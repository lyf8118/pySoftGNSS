/*================================================================================
 * Filename: corrGPUParallelQPSK.cu
 * Description: CUDA channel-parallel QPSK tracking correlator exposed through
 *              a small C ABI for Python ctypes.
 *
 * Authors: Yafeng Li (School of Automation, Beijing Information Science and Technology University)
 * Time: Feb, 20, 2026
 *
 * Notes:
 *   - Input/Output interface is kept compatible with the original implementation.
 *   - fileType = 1: real IF samples [I0 I1 I2 ...]
 *   - fileType = 2: complex interleaved IF samples [I0 Q0 I1 Q1 ...]
 *   - Output layout is 12 x channelCnt:
 *         [I_E; Q_E; I_P; Q_P; I_L; Q_L;
 *          pilot_I_E; pilot_Q_E; pilot_I_P; pilot_Q_P; pilot_I_L; pilot_Q_L]
 *================================================================================*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#define DLL_EXPORT extern "C" __declspec(dllexport)

// ============================ Configuration ====================================
#define THREADS_PER_BLOCK  256
#define CORR_NUMBER 6    // Data Early/Prompt/Late + Pilot Early/Prompt/Late
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
#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t err__ = (call);                                                               \
        if (err__ != cudaSuccess) {                                                               \
            printf("   %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__,                     \
                cudaGetErrorString(err__));                                                       \
            cleanup();                                                                            \
            return NULL;                                                                           \
        }                                                                                         \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                                       \
    do {                                                                                          \
        cudaError_t err__ = cudaPeekAtLastError();                                                \
        if (err__ != cudaSuccess) {                                                               \
            printf("   Kernel launch failed at %s:%d: %s\n", __FILE__, __LINE__,                 \
                cudaGetErrorString(err__));                                                       \
            cleanup();                                                                            \
            return NULL;                                                                           \
        }                                                                                         \
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
 * mixCarrReal()      : mix local carrier with real IF data on GPU
 * mixCarrComplex()   : mix local carrier with complex interleaved IF data on GPU
 * correlator()       : compute data/pilot Early / Prompt / Late correlations on GPU
 * cleanup()          : release persistent GPU / host memory
 * checkInputs()      : validate fixed C ABI values on the first call
 *------------------------------------------------------------------------------*/
__global__ void mixCarrReal(const short*, GPU_Complex*, int, float, float);
__global__ void mixCarrComplex(const short*, GPU_Complex*, int, float, float);
__global__ void correlator(GPU_Complex*, GPU_Complex*, const signed char* __restrict__, float, int, float, int, int);
void cleanup(void);

static int checkInputs(int fileType, int rawSignalLen, const short* rawSignal,
	const signed char* caCodeTable, int codeLen, int channelCnt,
	const double* remCarrPhase, const double* carrPhaseStep,
	const double* remCodePhase, const double* codePhaseStep,
	const int* startIndex, const int* chSampSize);

/* Persistent state -------------------------------------------------------------
 * d_pRawSignal        : device buffer for input IF signal block
 * d_ppCaCode          : per-channel device pointers to local code tables
 * d_ppBasebandSignal  : per-channel device buffers for carrier-wiped baseband
 * d_corrValues        : device pointer mapped to host correlation buffer
 * h_corrValues        : host mapped correlation buffer (Early/Prompt/Late)
 * streams             : one CUDA stream for each tracking channel
 * channelCnt          : number of tracking channels
 * settings            : cached receiver settings supplied through C ABI
 *------------------------------------------------------------------------------*/
static short * d_pRawSignal = NULL;
static signed char ** d_ppCaCode = NULL;
static GPU_Complex **d_ppBasebandSignal = NULL, * d_corrValues = NULL;
static std::complex<float>* h_corrValues = NULL;
static cudaStream_t* streams = NULL;
static size_t channelCnt = 0;
static Settings settings;
__constant__ float d_initCodePhase[CORR_NUMBER];

static double* corrValues = NULL;
static int initialized = 0;

/* Python ctypes entry point ---------------------------------------------------
 * caCodeTable is a C-order [channelCnt][2*codeLen] table, startIndex is
 * zero-based, and the returned persistent array contains 12 values per channel.
 *------------------------------------------------------------------------------*/
DLL_EXPORT double* corrEngine(int fileType, int rawSignalLenInput,
	double earlyLateSpc, short* rawSignal, signed char* caCodeTable,
	int codeLenInput, int channelCntInput, double* remCarrPhase,
	double* carrPhaseStep, double* remCodePhase, double* codePhaseStep,
	int* startIndex, int* chSampSize, int isDataReadInput)
{
	/* --------------- Declare all variables --------------- */
	static size_t codeLen;
	static size_t branchCodeLen;
	static int rawSignalLen;
	static size_t blockPerGrid;

	/* ================ Allocate host/device memory and initialize variables ================= */
	if (!initialized)
	{
		/* Validate the C ABI before persistent buffers are allocated. Keep this
		   check in the first-time initialization path so later calls remain on
		   the existing fast path. */
		if (!checkInputs(fileType, rawSignalLenInput, rawSignal, caCodeTable,
			codeLenInput, channelCntInput, remCarrPhase, carrPhaseStep,
			remCodePhase, codePhaseStep, startIndex, chSampSize)) {
			return NULL;
		}

		settings.trkMode = 1;
		settings.fileType = fileType;
		settings.earlyLateSpc = earlyLateSpc;

		/* ------ Get array dimensions ------
		 * branchCodeLen : number of chips in one guarded code branch
		 * codeLen       : total concatenated [dataCode pilotCode] length
		 * channelCnt    : number of tracking channels
		 * fileType      : 1 - real IF; 2 - complex interleaved IF
		 *----------------------------------- */
		branchCodeLen = (size_t)codeLenInput;
		codeLen = 2 * branchCodeLen;
		channelCnt = (size_t)channelCntInput;

		CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
		/* --------- Copy PRN codes and allocate per-channel GPU buffers ---------
		 * Each channel owns:
		 *   1) one device-side local code table
		 *   2) one device-side baseband buffer after carrier wipeoff
		 *------------------------------------------------------------------------ */
		d_ppCaCode = (signed char**)calloc(channelCnt, sizeof(signed char*));
		d_ppBasebandSignal = (GPU_Complex**)calloc(channelCnt, sizeof(GPU_Complex*));
		if (!d_ppCaCode || !d_ppBasebandSignal) {
			printf("   Failed to allocate host-side bookkeeping memory.\n");
			cleanup();
			return NULL;
		}

		size_t blksize = (size_t)chSampSize[0];
		for (int chInd = 0; chInd < channelCnt; chInd++) {
			signed char* caCode = &caCodeTable[(size_t)chInd * codeLen];
			// Allocate device global memory for the local code table
			CUDA_CHECK(cudaMalloc(&d_ppCaCode[chInd], sizeof(signed char) * codeLen));
			CUDA_CHECK(cudaMemcpy(d_ppCaCode[chInd], caCode, codeLen * sizeof(signed char),
				cudaMemcpyHostToDevice));
			// Allocate device global memory for the local complex baseband signal
			CUDA_CHECK(cudaMalloc(&d_ppBasebandSignal[chInd],
				sizeof(GPU_Complex) * (blksize + 100)));
		}
		// Grid size for carrier wipeoff kernels
		blockPerGrid = (blksize + 100 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		/* Each row of the combined QPSK code table is arranged as:
		     [dataCode(last,1..N,first), pilotCode(last,1..N,first)].
		   Keep only the Early/Prompt/Late offset here. MATLAB uses
		   ceil(codePhase +/- earlyLateSpc) + 1 on the guarded code table; the
		   CUDA kernel below implements the equivalent 0-based ceil index. The
		   data/pilot branch offset is added later as an integer index. */
		float initCodePhase[CORR_NUMBER] = {
			(float)(-settings.earlyLateSpc),
			0.0f,
			(float)(settings.earlyLateSpc),
			(float)(-settings.earlyLateSpc),
			0.0f,
			(float)(settings.earlyLateSpc)
		};
		CUDA_CHECK(cudaMemcpyToSymbol(d_initCodePhase, initCodePhase,
			sizeof(float) * CORR_NUMBER));

		/* --------- Input IF signal buffer --------- */
		rawSignalLen = rawSignalLenInput;
		CUDA_CHECK(cudaMalloc(&d_pRawSignal, (size_t)rawSignalLen * sizeof(short)));

		/* --------- Correlation values ---------
		 * Host mapped pinned memory is used so the GPU can write correlation
		 * results directly to a host-visible buffer without an extra copy.
		 *---------------------------------------- */
		CUDA_CHECK(cudaHostAlloc(&h_corrValues,
			CORR_NUMBER * channelCnt * sizeof(std::complex<float>), cudaHostAllocMapped));
		CUDA_CHECK(cudaHostGetDevicePointer(&d_corrValues, h_corrValues, 0));

		/* -------- Output allocation ---------
		 * Each channel contributes 12 values:
		 * data   [I_E Q_E I_P Q_P I_L Q_L]
		 * pilot  [I_E Q_E I_P Q_P I_L Q_L].
		 *------------------------------------- */
		corrValues = (double*)malloc((size_t)(2 * CORR_NUMBER) * channelCnt * sizeof(double));

		/* ------- Allocate CUDA streams ------- */
		streams = (cudaStream_t*)calloc(channelCnt, sizeof(cudaStream_t));
		if (corrValues == NULL || streams == NULL) {
			cleanup();
			return NULL;
		}
		for (int chInd = 0; chInd < channelCnt; chInd++) {
			CUDA_CHECK(cudaStreamCreate(&streams[chInd]));
		}

		initialized = 1;
		printf("   CUDA initialized ...\n");
		printf("   Channel-parallel tracking with the GPU-accelerated correlator developed by Yafeng Li.\n");
	}

	/* ========================= GPU correlator implementation =========================
	 * 1) Upload a new IF block when requested
	 * 2) For each channel:
	 *      - mix carrier to form complex baseband samples
	 *      - perform data/pilot Early/Prompt/Late correlations
	 * 3) Synchronize all streams
	 * 4) Pack results into the persistent Python output array
	 *=============================================================================== */
	/* Upload IF data when needed */
	if (isDataReadInput == 1) {
		CUDA_CHECK(cudaMemcpy(d_pRawSignal, rawSignal,
			(size_t)rawSignalLen * sizeof(short), cudaMemcpyHostToDevice));
	}

	/* --------- Per-channel GPU processing --------- */
	for (int chInd = 0; chInd < channelCnt; chInd++) {
		double wrapped = fmod(remCodePhase[chInd], (double)branchCodeLen);
		int codePhaseBase = (int)floor(wrapped);
		float remCodePhaseFrac = (float)(wrapped - (double)codePhaseBase);
		// 1) Carrier wipe-off to form local baseband samples.
		if (settings.fileType == 1) {
			mixCarrReal<<<blockPerGrid, THREADS_PER_BLOCK, 0, streams[chInd]>>>(
				d_pRawSignal + startIndex[chInd], d_ppBasebandSignal[chInd],
				chSampSize[chInd], (float)remCarrPhase[chInd], (float)carrPhaseStep[chInd]);
		} else {
			mixCarrComplex<<<blockPerGrid, THREADS_PER_BLOCK, 0, streams[chInd]>>>(
				d_pRawSignal + startIndex[chInd] * 2, d_ppBasebandSignal[chInd],
				chSampSize[chInd], (float)remCarrPhase[chInd], (float)carrPhaseStep[chInd]);
		}
		CUDA_KERNEL_CHECK();
		// 2) Data / pilot Early / Prompt / Late correlation.
		correlator<<<CORR_NUMBER, THREADS_PER_BLOCK, 0, streams[chInd]>>>(
			d_corrValues + CORR_NUMBER * chInd, d_ppBasebandSignal[chInd],
			d_ppCaCode[chInd], (float)codePhaseStep[chInd], codePhaseBase,
			remCodePhaseFrac, chSampSize[chInd], (int)branchCodeLen);
		CUDA_KERNEL_CHECK();
	}

	/* --------- Wait for all streams to finish before reading mapped host memory --------- */
	CUDA_CHECK(cudaDeviceSynchronize());
	// -------------------- Pack output for Python ---------------------------
	// Output order per channel:
	// [I_E, Q_E, I_P, Q_P, I_L, Q_L,
	//  pilot_I_E, pilot_Q_E, pilot_I_P, pilot_Q_P, pilot_I_L, pilot_Q_L].
	for (int chInd = 0; chInd < channelCnt; chInd++) {
		int correIndex = CORR_NUMBER * chInd;
		for (int ind = 0; ind < CORR_NUMBER; ind++) {
			corrValues[12 * chInd + ind * 2] = (double)h_corrValues[correIndex + ind].real();
			corrValues[12 * chInd + ind * 2 + 1] = (double)h_corrValues[correIndex + ind].imag();
		}
	}
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
 * mixCarrReal
 * ----------
 * Mix a real-valued IF sequence with a complex carrier replica:
 *     bb[n] = raw[n] * exp(-j*(remCarrPhase + n*carrPhaseStep))
 */
__global__ void mixCarrReal(const short* d_pRawSignal, GPU_Complex* d_BasebandSignal,int blksize, float remCarrPhase,float carrPhaseStep)
{
	// CUDA version of floating point NCO and vector dot product integrated
	
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < blksize; index += blockDim.x * gridDim.x){
		float sin, cos;
		__sincosf(remCarrPhase + index * carrPhaseStep, &sin, &cos);
		d_BasebandSignal[index] = GPU_Complex(cos, -sin) * (float)d_pRawSignal[index];
	}
}

/*
 * mixCarrComplex
 * -------------
 * Mix a complex interleaved IF sequence [I,Q,I,Q,...] with the same local carrier:
 *     bb[n] = (I[n] + jQ[n]) * exp(-j*(remCarrPhase + n*carrPhaseStep))
 */
__global__ void mixCarrComplex(const short* d_pRawSignal, GPU_Complex* d_BasebandSignal,
	int blksize, float remCarrPhase, float carrPhaseStep)
{
	// CUDA version of floating point NCO and vector dot product integrated
	const short2* d_pRawSignalIQ = reinterpret_cast<const short2*>(d_pRawSignal);
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < blksize; index += blockDim.x * gridDim.x) {
		float sin, cos;
		short2 rawIQ = d_pRawSignalIQ[index];
		__sincosf(remCarrPhase + index * carrPhaseStep, &sin, &cos);
		d_BasebandSignal[index] = GPU_Complex(cos, -sin) *
			GPU_Complex((float)rawIQ.x, (float)rawIQ.y);
	}
}

/*
 * correlator
 * ----------
 * Compute QPSK data/pilot Early / Prompt / Late correlations.
 * blockIdx.x selects the correlator branch:
 *   vec = 0..2 -> data  Early / Prompt / Late
 *   vec = 3..5 -> pilot Early / Prompt / Late
 *
 * Each thread accumulates a strided partial sum, then a shared-memory tree reduction
 * combines them to obtain one complex correlation value per branch.
 */
__global__ void correlator(GPU_Complex* d_corrValues, GPU_Complex* d_BasebandSignal, const signed char* __restrict__ d_caCodeTable,
    float codePhaseStep, int codePhaseBase, float remCodePhase, int blksize, int branchCodeLen)
{
    /* Each block computes one QPSK correlator branch:
         block 0..2 -> data Early / Prompt / Late
         block 3..5 -> pilot Early / Prompt / Late */
	__shared__ GPU_Complex accumResult[ACCUM_N];

	for (int vec = blockIdx.x; vec < CORR_NUMBER; vec += gridDim.x)
	{
        int branchOffset = (vec / 3) * branchCodeLen;
        float initPhase = d_initCodePhase[vec] + remCodePhase;

		for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
		{
			GPU_Complex sumIQ(0.0f, 0.0f);

			//float code_phase;
			for (int pos = iAccum; pos < blksize; pos += ACCUM_N) {
				// 1.resample local code for the current shift
				float localPhase = codePhaseStep * __int2float_rd(pos) + initPhase;
				int chipIndex = branchOffset + codePhaseBase + __float2int_ru(localPhase);

				// 2.correlate
				sumIQ.multiply_acc(d_BasebandSignal[pos], (float)__ldg(&d_caCodeTable[chipIndex]));
			}
			accumResult[iAccum] = sumIQ;
		}

		// Tree reduction in shared memory. ACCUM_N must be a power of two.
		for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1) {
			__syncthreads();
			for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
				accumResult[iAccum] += accumResult[stride + iAccum];}
		}

		if (threadIdx.x == 0) {
			d_corrValues[vec] = accumResult[0];}

	}
}

/* checkInputs ----------------------------------------------------------------
 * Validate fixed C ABI values before persistent buffers are allocated.
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
		printf("   Invalid input arguments for corrGPUParallelQPSK.\n");
		return 0;
	}
	return 1;
}

/* cleanup --------------------------------------------------------------------
 * Release all persistent device and host-side resources when Python calls
 * corrEngineFree().
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
		if (d_ppBasebandSignal != NULL && d_ppBasebandSignal[chInd] != NULL) {
			(void)cudaFree(d_ppBasebandSignal[chInd]);
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
	if (corrValues != NULL) { free(corrValues); corrValues = NULL; }

	if (streams != NULL)  { free(streams); streams = NULL; }
	if (d_ppCaCode != NULL)  { free(d_ppCaCode); d_ppCaCode = NULL; }
	if (d_ppBasebandSignal != NULL)  { free(d_ppBasebandSignal); d_ppBasebandSignal = NULL; }
	initialized = 0;
	channelCnt = 0;
}
