/*================================================================================
 * Filename: corrGpuParallelFusedBPSK.cu
 * Description: CUDA DLL implementation of GPU-assisted GNSS tracking correlator for Python ctypes.
 *
 * Authors: Yafeng Li (School of Automation, Beijing Information Science and Technology University)
 * Time: Feb, 20, 2026
 *
 * Notes:
 *   - Core processing is kept compatible with the original implementation.
 *   - fileType = 1: real IF samples [I0 I1 I2 ...]
 *   - fileType = 2: complex interleaved IF samples [I0 Q0 I1 Q1 ...]
 *   - Output layout is 6 x channelCnt:
 *         [IE; QE; IP; QP; IL; QL]
 *================================================================================*/

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#define DLL_EXPORT extern "C" __declspec(dllexport)

// ============================ Configuration ====================================
#define THREADS_PER_BLOCK  256
#define CORR_NUMBER 3    // Early / Prompt / Late    
#define ACCUM_N 256

/* Settings struct -------------------------------------------------------------
 * trkMode        : 0 - channel-serial tracking; 1 - channel-parallel tracking
 * fileType       : 1 - real samples; 2 - complex samples
 * earlyLateSpc   : half of the early - late code correlation spacing (chips)
 *------------------------------------------------------------------------------*/
typedef struct {
	int trkMode;
	int fileType;
    double earlyLateSpc;
} Settings;

// ============================ CUDA helpers =====================================
#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t err__ = (call);                                                              \
        if (err__ != cudaSuccess) {                                                              \
            printf("   %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__,                     \
                cudaGetErrorString(err__));                                                      \
            cleanup();                                                                           \
            return NULL;                                                                         \
        }                                                                                        \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                                      \
    do {                                                                                         \
        cudaError_t err__ = cudaPeekAtLastError();                                               \
        if (err__ != cudaSuccess) {                                                              \
            printf("   Kernel launch failed at %s:%d: %s\n", __FILE__, __LINE__,                 \
                cudaGetErrorString(err__));                                                      \
            cleanup();                                                                           \
            return NULL;                                                                         \
        }                                                                                        \
    } while (0)

/* ========== GPU internal data types for complex numbers ========== */
struct GPU_Complex
{
	float r;
	float i;
	__host__ __device__ GPU_Complex() : r(0.0f), i(0.0f) {};

	// Constructor for a complex number with real amd imag parts of float
    __host__ __device__ GPU_Complex(float realPart, float imagPart) : r(realPart), i(imagPart) {}

	// Magnitude of a complex number
	__host__ __device__ float magnitude2() { return r * r + i * i; }

	__device__ GPU_Complex operator*(const GPU_Complex& a) {
		return GPU_Complex(__fmul_rn(r, a.r) - __fmul_rn(i, a.i), __fmul_rn(i, a.r) + __fmul_rn(r, a.i));
	}

	__device__ GPU_Complex operator*(const float& a) {
		return GPU_Complex(__fmul_rn(r, a), __fmul_rn(i, a));
	}

	__host__ __device__ GPU_Complex operator+(const GPU_Complex& a) {
		return GPU_Complex(r + a.r, i + a.i);
	}

	__host__ __device__ void operator+=(const GPU_Complex& a) {
		r += a.r;
		i += a.i;
	}

    // Accumulate a * b into *this, where b is complex.
	__device__ void multiply_acc(const GPU_Complex& a, const GPU_Complex& b) {
		//real part
		r = __fmaf_rn(a.r, b.r, r);
		r = __fmaf_rn(-a.i, b.i, r);
		//imag part
		i = __fmaf_rn(a.i, b.r, i);
		i = __fmaf_rn(a.r, b.i, i);
	}

    // Accumulate a * b into *this, where b is real.
	__device__ void multiply_acc(const GPU_Complex& a, const float& b) {
		//real part
		r = __fmaf_rn(a.r, b, r);
		//imag part
		i = __fmaf_rn(a.i, b, i);
	}
};

/* Function declaration ---------------------------------------------------------
 * fusedCorrReal()    : fused carrier wipeoff + Early/Prompt/Late correlation for real IF data
 * fusedCorrComplex() : fused carrier wipeoff + Early/Prompt/Late correlation for complex IF data
 * cleanup()          : release persistent GPU / host memory
 * checkInputs()      : validate the fixed interface values on the first call
 *------------------------------------------------------------------------------*/
__global__ void fusedCorrReal(GPU_Complex*, const short*, const signed char* __restrict__, float, int, float, float, int, float, float);
__global__ void fusedCorrComplex(GPU_Complex*, const short*, const signed char* __restrict__, float, int, float, float, int, float, float);
void cleanup(void);
static int checkInputs(int fileType, int rawSignalLen, const short* rawSignal,
	const signed char* caCodeTable, int codeLen, int channelCnt,
	const double* remCarrPhase, const double* carrPhaseStep,
	const double* remCodePhase, const double* codePhaseStep,
	const int* startIndex, const int* chSampSize);

/* Persistent state -------------------------------------------------------------
 * d_pRawSignal        : device buffer for input IF signal block (int16 samples)
 * d_ppCaCode          : per-channel device pointers to local code tables
 * d_corrValues        : device pointer mapped to host correlation buffer
 * h_corrValues        : host mapped correlation buffer (Early/Prompt/Late)
 * streams             : one CUDA stream for each tracking channel
 * channelCnt          : number of tracking channels
 * settings            : cached receiver settings supplied through C ABI
 *------------------------------------------------------------------------------*/
static short * d_pRawSignal;
static signed char ** d_ppCaCode;
static GPU_Complex * d_corrValues;
static std::complex<float>* h_corrValues;
static cudaStream_t* streams;
static size_t channelCnt;
static Settings settings;

/* Python-only persistent state -------------------------------------------------
 * corrValues          : ctypes-visible [IE,QE,IP,QP,IL,QL] output array
 * initialized         : initialization flag shared with cleanup()
 *------------------------------------------------------------------------------*/
static double* corrValues;
static int initialized = 0;

/* The gateway function --------------------------------------------------------
 * Python ctypes entry point.
 * Args   : fileType       I   1 - real; 2 - interleaved [I,Q] int16 samples
 *          rawSignalLen   I   physical number of int16 elements in rawSignal
 *          earlyLateSpc   I   half early-late spacing (chips)
 *          rawSignal      I   shared IF block
 *          caCodeTable    I   int8 C-order table [channelCnt][codeLen]
 *          codeLen        I   guarded-code elements per channel
 *          channelCnt     I   active tracking-channel count
 *          remCarrPhase   I   residual carrier phases (rad)
 *          carrPhaseStep  I   carrier phase increments (rad/sample)
 *          remCodePhase   I   residual code phases (chips)
 *          codePhaseStep  I   code phase increments (chips/sample)
 *          startIndex     I   zero-based logical sample offsets in rawSignal
 *          chSampSize     I   logical sample count for each channel
 *          isDataRead     I   1 uploads new rawSignal data; 0 reuses it
 * Return : persistent channel-major double array with six values per channel,
 *          [IE,QE,IP,QP,IL,QL], or NULL on invalid input/CUDA failure. The
 *          pointer remains valid until corrEngineFree(); each successful
 *          corrEngine() call overwrites the array contents.
 *------------------------------------------------------------------------------*/
DLL_EXPORT double* corrEngine(int fileType, int rawSignalLenInput,
	double earlyLateSpc, short* rawSignal, signed char* caCodeTable,
	int codeLenInput, int channelCntInput, double* remCarrPhase,
	double* carrPhaseStep, double* remCodePhase, double* codePhaseStep,
	int* startIndex, int* chSampSize, int isDataReadInput)
{
	/* --------------- Declare all variables --------------- */
	static size_t codeLen;
	static int isDataRead;
	static int rawSignalLen;
	/* ================ Allocate host/device memory and initialize variables ================= */
	if (!initialized)
	{
		if (!checkInputs(fileType, rawSignalLenInput, rawSignal, caCodeTable,
			codeLenInput, channelCntInput, remCarrPhase, carrPhaseStep,
			remCodePhase, codePhaseStep, startIndex, chSampSize)) {
			return NULL;
		}

		settings.trkMode = 1;
		// file type: real or complex
		settings.fileType = fileType;

		/* ------ Get array dimensions ------
		 * codeLen    : number of elements in each C-order code-table row
		 * channelCnt : number of tracking channels
		 * fileType   : 1 - real IF; 2 - complex interleaved IF
		 *------------------------------------ */
		//Size of the PRN code chips
		codeLen = (size_t)codeLenInput;
		//Channel count
		channelCnt = (size_t)channelCntInput;

		/* --------- Copy PRN codes to per-channel device buffers ---------
		 * Each channel owns one device-side local code table.
		 * The carrier wipeoff result is no longer staged in a separate
		 * baseband buffer; it is consumed directly inside the fused kernel.
		 *------------------------------------------------------------------------ */
		// PRN codes memory
		d_ppCaCode = (signed char**)calloc(channelCnt, sizeof(signed char*));
		if (d_ppCaCode == NULL) {
			cleanup();
			return NULL;
		}

		for (int chInd = 0; chInd < channelCnt; chInd++)
		{
			// Python stores one channel per contiguous C-order row.
			size_t index = (size_t)chInd * codeLen;
			signed char* caCode = &caCodeTable[index];
			//malloc device global memory for caCodeTable
			CUDA_CHECK(cudaMalloc(&d_ppCaCode[chInd], sizeof(signed char) * codeLen));
			CUDA_CHECK(cudaMemcpy(d_ppCaCode[chInd], caCode, codeLen * sizeof(signed char), cudaMemcpyHostToDevice));
		}

		/* --------- For input IF signal --------- */
		// malloc device global memory for RawSignal
		// rawSignalLen is the physical int16 element count for both file types.
		rawSignalLen = rawSignalLenInput;
		CUDA_CHECK(cudaMalloc(&d_pRawSignal, (size_t)rawSignalLen * sizeof(short)));

		/* --------- Correlation values ---------
		 * Host mapped pinned memory is used so the GPU can write correlation
		 * results directly to a host-visible buffer without an extra copy.
		 *---------------------------------------- */
		//malloc device host mapped page-locked memory for corrValues
		CUDA_CHECK(cudaHostAlloc(&h_corrValues, CORR_NUMBER * channelCnt * sizeof(std::complex<float>), cudaHostAllocMapped));
		// Get device pointer from host memory
		CUDA_CHECK(cudaHostGetDevicePointer(&d_corrValues, h_corrValues, 0));

		/* Persistent ctypes output array */
		corrValues = (double*)malloc((size_t)6 * channelCnt * sizeof(double));
		if (corrValues == NULL) {
			cleanup();
			return NULL;
		}

		/* ------- Allocate CUDA streams ------- */
		streams = (cudaStream_t*)calloc(channelCnt, sizeof(cudaStream_t));
		if (streams == NULL) {
			cleanup();
			return NULL;
		}
		for (int chInd = 0; chInd < channelCnt; chInd++) {
			CUDA_CHECK(cudaStreamCreate(&streams[chInd]));
		}

		initialized = 1;
		printf("   CUDA initialized ...\n");
		printf("   Channel-parallel tracking with the GPU-accelerated correlator developed by Yafeng Li.\n");
	} // for initialization

	/* ----------------- Get input values ------------------- */
	isDataRead = isDataReadInput;
	settings.earlyLateSpc = earlyLateSpc;

    /* ========================= GPU correlator implementation =========================
     * 1) Upload a new IF block when requested
     * 2) For each channel:
     *      - run one fused kernel that generates the carrier replica,
     *        forms the local code for Early/Prompt/Late, and accumulates
     *        the three correlations inside one block
     * 3) Synchronize all streams
     * 4) Pack results into the Python output array
     *=============================================================================== */
	/* Upload IF data when needed */
	if (isDataRead == 1) {
		CUDA_CHECK(cudaMemcpy(d_pRawSignal, rawSignal, (size_t)rawSignalLen * sizeof(short), cudaMemcpyHostToDevice));
	}

	/* --------- Per-channel GPU processing --------- */
	for (int chInd = 0; chInd < channelCnt; chInd++)
	{
		float initCodePhase = 1.0f - (float)settings.earlyLateSpc + (float)remCodePhase[chInd];
		// 1) Run one fused block per channel.
		if (settings.fileType == 1)
		{ // for real data
			fusedCorrReal<<<1, THREADS_PER_BLOCK, 0, streams[chInd]>>>(
				d_corrValues + CORR_NUMBER * chInd,
				d_pRawSignal + startIndex[chInd],
				d_ppCaCode[chInd],
				(float)settings.earlyLateSpc,
				(int)codeLen,
				(float)codePhaseStep[chInd],
				initCodePhase,
				chSampSize[chInd],
				(float)remCarrPhase[chInd],
				(float)carrPhaseStep[chInd]);
		}
		else if (settings.fileType == 2)
		{ // for complex data
			fusedCorrComplex<<<1, THREADS_PER_BLOCK, 0, streams[chInd]>>>(
				d_corrValues + CORR_NUMBER * chInd,
				d_pRawSignal + startIndex[chInd] * 2,
				d_ppCaCode[chInd],
				(float)settings.earlyLateSpc,
				(int)codeLen,
				(float)codePhaseStep[chInd],
				initCodePhase,
				chSampSize[chInd],
				(float)remCarrPhase[chInd],
				(float)carrPhaseStep[chInd]);
		}
		CUDA_KERNEL_CHECK();
	}

	/* --------- Wait for all streams to finish before reading mapped host memory --------- */
	CUDA_CHECK(cudaDeviceSynchronize());

	// -------------------- Pack output for Python ---------------------------
	// Output order per channel: [IE, QE, IP, QP, IL, QL]^T.
	for (int chInd = 0; chInd < channelCnt; chInd++) {
		int correIndex = CORR_NUMBER * chInd;
		for (int ind = 0; ind < CORR_NUMBER; ind++) {
			corrValues[6 * chInd + ind * 2] = (double)h_corrValues[correIndex + ind].real();
			corrValues[6 * chInd + ind * 2 + 1] = (double)h_corrValues[correIndex + ind].imag();
		}
	}
	// Display the last error for debug
	//CUDA_CHECK(cudaGetLastError());
	return corrValues;
}

/* corrEngineFree -------------------------------------------------------------
 * Explicitly release persistent resources from the Python close/finally path.
 * It is safe to call repeatedly.
 *------------------------------------------------------------------------------*/
DLL_EXPORT void corrEngineFree(void)
{
	cleanup();
}

// ============================ CUDA kernels =====================================
/*
 * fusedCorrReal
 * -------------
 * Fused carrier wipeoff and Early / Prompt / Late correlation for real IF data.
 * One block handles one tracking channel. Each thread generates the local carrier
 * and the three local code branches for its strided samples, then a shared-memory
 * reduction combines the per-thread partial sums into [E, P, L].
 */
__global__ void fusedCorrReal(GPU_Complex* d_corrValues, const short* d_pRawSignal, const signed char* __restrict__ d_caCodeTable,
	float earlyLateSpc, int codeLen, float codePhaseStep, float initCodePhase, int blksize,
	float remCarrPhase, float carrPhaseStep)
{
	__shared__ GPU_Complex accumResult[CORR_NUMBER][ACCUM_N];
	float initPhase[CORR_NUMBER];
	#pragma unroll
	for (int vec = 0; vec < CORR_NUMBER; ++vec) {
		initPhase[vec] = initCodePhase + earlyLateSpc * vec;
	}

	for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
	{
		GPU_Complex sumIQ[CORR_NUMBER] = { GPU_Complex(0.0f, 0.0f), GPU_Complex(0.0f, 0.0f), GPU_Complex(0.0f, 0.0f) };

		for (int pos = iAccum; pos < blksize; pos += ACCUM_N) {
			float sinVal, cosVal;
			__sincosf(remCarrPhase + pos * carrPhaseStep, &sinVal, &cosVal);
			GPU_Complex baseband = GPU_Complex(cosVal, -sinVal) * (float)d_pRawSignal[pos];

			#pragma unroll
			for (int vec = 0; vec < CORR_NUMBER; ++vec) {
				int chipIndex = __float2int_rd(fmodf(codePhaseStep * __int2float_rd(pos) + initPhase[vec], codeLen));
				sumIQ[vec].multiply_acc(baseband, (float)__ldg(&d_caCodeTable[chipIndex]));
			}
		}

		#pragma unroll
		for (int vec = 0; vec < CORR_NUMBER; ++vec) {
			accumResult[vec][iAccum] = sumIQ[vec];
		}
	}

	// Tree reduction in shared memory. ACCUM_N must be a power of two.
	for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1) {
		__syncthreads();
		for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
			#pragma unroll
			for (int vec = 0; vec < CORR_NUMBER; ++vec) {
				accumResult[vec][iAccum] += accumResult[vec][stride + iAccum];
			}
		}
	}

	if (threadIdx.x == 0) {
		#pragma unroll
		for (int vec = 0; vec < CORR_NUMBER; ++vec) {
			d_corrValues[vec] = accumResult[vec][0];
		}
	}
}

/*
 * fusedCorrComplex
 * ----------------
 * Fused carrier wipeoff and Early / Prompt / Late correlation for complex IF data.
 * The input is interleaved as [I0, Q0, I1, Q1, ...]. The carrier and three code
 * branches are generated inside the same block, then reduced to one [E, P, L]
 * output tuple for the channel. Complex raw samples are loaded as short2 so I/Q
 * are fetched together with one vector load.
 */
__global__ void fusedCorrComplex(GPU_Complex* d_corrValues, const short* d_pRawSignal, const signed char* __restrict__ d_caCodeTable,
	float earlyLateSpc, int codeLen, float codePhaseStep, float initCodePhase, int blksize,
	float remCarrPhase, float carrPhaseStep)
{
	__shared__ GPU_Complex accumResult[CORR_NUMBER][ACCUM_N];
	const short2* d_pRawSignalIQ = reinterpret_cast<const short2*>(d_pRawSignal);
	float initPhase[CORR_NUMBER];
	#pragma unroll
	for (int vec = 0; vec < CORR_NUMBER; ++vec) {
		initPhase[vec] = initCodePhase + earlyLateSpc * vec;
	}

	for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
	{
		GPU_Complex sumIQ[CORR_NUMBER] = { GPU_Complex(0.0f, 0.0f), GPU_Complex(0.0f, 0.0f), GPU_Complex(0.0f, 0.0f) };

		for (int pos = iAccum; pos < blksize; pos += ACCUM_N) {
			float sinVal, cosVal;
			__sincosf(remCarrPhase + pos * carrPhaseStep, &sinVal, &cosVal);
			short2 rawIQ = d_pRawSignalIQ[pos];
			GPU_Complex baseband = GPU_Complex(cosVal, -sinVal) *
				GPU_Complex((float)rawIQ.x, (float)rawIQ.y);

			#pragma unroll
			for (int vec = 0; vec < CORR_NUMBER; ++vec) {
				int chipIndex = __float2int_rd(fmodf(codePhaseStep * __int2float_rd(pos) + initPhase[vec], codeLen));
				sumIQ[vec].multiply_acc(baseband, (float)__ldg(&d_caCodeTable[chipIndex]));
			}
		}

		#pragma unroll
		for (int vec = 0; vec < CORR_NUMBER; ++vec) {
			accumResult[vec][iAccum] = sumIQ[vec];
		}
	}

	// Tree reduction in shared memory. ACCUM_N must be a power of two.
	for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1) {
		__syncthreads();
		for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
			#pragma unroll
			for (int vec = 0; vec < CORR_NUMBER; ++vec) {
				accumResult[vec][iAccum] += accumResult[vec][stride + iAccum];
			}
		}
	}

	if (threadIdx.x == 0) {
		#pragma unroll
		for (int vec = 0; vec < CORR_NUMBER; ++vec) {
			d_corrValues[vec] = accumResult[vec][0];
		}
	}
}

/* checkInputs ----------------------------------------------------------------
 * Validate the fixed interface values once, before persistent buffers are
 * allocated. Later calls reuse the initialized configuration.
 * Args   : fixed dimensions and array pointers supplied to the first
 *          corrEngine() call
 * Return : 1 for valid input; 0 otherwise
 *------------------------------------------------------------------------------*/
static int checkInputs(int fileType, int rawSignalLen,
	const short* rawSignal, const signed char* caCodeTable,
	int codeLen, int channelCnt,
	const double* remCarrPhase, const double* carrPhaseStep,
	const double* remCodePhase, const double* codePhaseStep,
	const int* startIndex, const int* chSampSize)
{
	if ((fileType != 1 && fileType != 2) || rawSignalLen <= 0 ||
		codeLen <= 0 || channelCnt <= 0 || rawSignal == NULL ||
		caCodeTable == NULL || remCarrPhase == NULL ||
		carrPhaseStep == NULL || remCodePhase == NULL ||
		codePhaseStep == NULL || startIndex == NULL || chSampSize == NULL) {
		printf("   Invalid input arguments for corrGPUParallelFusedBPSK.\n");
		return 0;
	}
	return 1;
}

/* cleanup --------------------------------------------------------------------
 * Release all persistent device and host-side resources when corrEngineFree()
 * is called by Python.
 * Return : None
 *------------------------------------------------------------------------------*/
void cleanup(void)
{
	printf("   CUDA is terminating, destroying allocated memory ...\n");

	/* --------- Free device-side memory --------- */
	for (size_t chInd = 0; chInd < channelCnt; ++chInd) {
		if (d_ppCaCode != NULL && d_ppCaCode[chInd] != NULL) {
			(void)cudaFree(d_ppCaCode[chInd]);
		}
		if (streams != NULL && streams[chInd] != NULL) {
			(void)cudaStreamDestroy(streams[chInd]);
		}
	}

	if (d_pRawSignal != NULL) {
		(void)cudaFree(d_pRawSignal);
		d_pRawSignal = NULL;
	}
	/* --------- Free host-side memory --------- */
	if (h_corrValues != NULL) {
		(void)cudaFreeHost(h_corrValues);
		h_corrValues = NULL;
		d_corrValues = NULL;
	}
	if (corrValues != NULL) {
		free(corrValues);
		corrValues = NULL;
	}
	if (streams != NULL) { free(streams); streams = NULL; }
	if (d_ppCaCode != NULL) { free(d_ppCaCode); d_ppCaCode = NULL; }

	initialized = 0;
	channelCnt = 0;
}
