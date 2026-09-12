/*================================================================================
 * Filename: corrGPUSerialBPSK.cu
 * Description: GPU-assisted single-channel tracking correlator exposed
 *              through a small C ABI for Python ctypes.
 *
 * Authors: Yafeng Li (School of Automation, Beijing Information Science and Technology University)
 * Time: Apr, 21, 2026
 *
 * The core processing stages are intentionally kept aligned with the
 * GPU channel-parallel correlator:
 *   1) mixCarrReal() / mixCarrComplex()
 *   2) correlator()
 *
 * This file processes one channel and one code period per call. The first
 * corrEngine() call allocates reusable CUDA buffers; corrEngineFree() releases
 * them after channel-serial tracking has finished.
 *===============================================================================*/

#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#define DLL_EXPORT extern "C" __declspec(dllexport)

// ============================ Configuration ====================================
#define THREADS_PER_BLOCK  256
#define CORR_NUMBER 3    // Early / Prompt / Late
#define ACCUM_N 256

/* Settings struct -------------------------------------------------------------
 * This structure stores the receiver settings cached by the GPU correlator.
 * fileType : 1 - real samples; 2 - interleaved complex samples
 * earlyLateSpc remains a per-call corrEngine() argument.
 *------------------------------------------------------------------------------*/
typedef struct
{
    int fileType;       /* 1 - real samples; 2 - complex samples */
    double earlyLateSpc;
} Settings;
// ============================ CUDA helpers =====================================
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err__ = (call);                                                             \
        if (err__ != cudaSuccess) {                                                             \
            printf("   %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__,                    \
                cudaGetErrorString(err__));                                                     \
            cleanup();                                                                          \
            return NULL;                                                                        \
        }                                                                                       \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                                     \
    do {                                                                                        \
        cudaError_t err__ = cudaPeekAtLastError();                                              \
        if (err__ != cudaSuccess) {                                                             \
            printf("   Kernel launch failed at %s:%d: %s\n", __FILE__, __LINE__,                \
                cudaGetErrorString(err__));                                                     \
            cleanup();                                                                          \
            return NULL;                                                                        \
        }                                                                                       \
    } while (0)



/* ========== GPU internal data types for complex numbers ========== */
struct GPU_Complex
{
    float r;
    float i;
    __host__ __device__ GPU_Complex() : r(0.0f), i(0.0f) {}

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
 * correlator()       : compute Early / Prompt / Late correlations on GPU
 * cleanup()          : release persistent device and host buffers
 * checkInputs()      : validate the fixed interface values on the first call
 *------------------------------------------------------------------------------*/
__global__ void mixCarrReal(const short*, GPU_Complex*, int, float, float);
__global__ void mixCarrComplex(const short*, GPU_Complex*, int, float, float);
__global__ void correlator(GPU_Complex*, GPU_Complex*, const signed char* __restrict__, float, int, float, float, int);
void cleanup(void);
static int checkInputs(int fileType, int rawSignalLenIn, const short* rawSignal,
    const signed char* caCode, int codeLen);

/* Persistent state -------------------------------------------------------------
 * d_pRawSignal     : device buffer for one IF signal block (int16 samples)
 * d_caCode         : device buffer for one local code table
 * d_BasebandSignal : device buffer for carrier-wiped baseband samples
 * d_corrValues     : device pointer mapped to host correlation buffer
 * h_corrValues     : host mapped correlation buffer (Early/Prompt/Late)
 * settings         : cached receiver settings parsed from C ABI arguments
 * blockPerGrid     : grid size used by the carrier wipeoff kernels
 * initialized      : first-call allocation flag
 *------------------------------------------------------------------------------*/
static short* d_pRawSignal = NULL;
static signed char* d_caCode = NULL;
static GPU_Complex* d_BasebandSignal = NULL;
static std::complex<float>* h_corrValues = NULL;
static GPU_Complex* d_corrValues = NULL;

static Settings settings;
static int initialized = 0;

/* Python ctypes entry point ---------------------------------------------------
 * Args   : fileType       I   1 - real samples; 2 - interleaved I/Q samples
 *          rawSignalLenIn I   number of int16 elements in rawSignal
 *          earlyLateSpc   I   half early-late spacing (chips)
 *          rawSignal      I   int16 IF samples
 *          caCode         I   int8 guarded code [last, one period, first]
 *          codeLen        I   number of elements in caCode
 *          remCarrPhase   I   residual carrier phase (rad)
 *          carrPhaseStep  I   carrier phase increment (rad/sample)
 *          remCodePhase   I   residual code phase (chips)
 *          codePhaseStep  I   code phase increment (chips/sample)
 *          PRN            I   current satellite PRN for device-code caching
 * Return : pointer to static [IE QE IP QP IL QL], or NULL when first-call
 *          validation, memory allocation, or CUDA processing fails. The
 *          pointer is overwritten by the next corrEngine() call.
 *------------------------------------------------------------------------------*/
DLL_EXPORT double* corrEngine(int fileType, int rawSignalLenIn, double earlyLateSpc,
    short* rawSignal, signed char* caCode, int codeLen,
    double remCarrPhase, double carrPhaseStep, double remCodePhase,
    double codePhaseStep, int inPRN)
{
    /* --------------- Declare all variables --------------- */
    size_t rawSignalLen;
    int blksize;
    static double corrValues[CORR_NUMBER * 2] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    static int PRN = 0;
    static size_t blockPerGrid = 0;

    if (!initialized)
    {
        /* Validate the interface before the first allocation. Later calls stay
           on the same fast path and reuse the initialized configuration. */
        // gateway check
        if (!checkInputs(fileType, rawSignalLenIn, rawSignal, caCode, codeLen)) {
            return NULL;
        }

        // Basic setting
        settings.fileType = fileType;

        /* --------------- Find array dimensions ---------------
         * codeLen      : number of guarded-code elements
         * rawSignalLen : number of int16 entries in the first signal block
         * blksize      : logical samples processed by the kernels; complex
         *                input contains two int16 entries per logical sample
         *------------------------------------------------------ */
        rawSignalLen = (size_t)rawSignalLenIn;
        blksize = (settings.fileType == 1) ? (int)rawSignalLen : (int)(rawSignalLen / 2);

        /* --------------------- Allocate memory ---------------------
         * Add a small safety margin because blksize can vary slightly
         * across tracking loops as code/carrier NCOs change.
         *------------------------------------------------------------ */
        rawSignalLen = rawSignalLen + ((settings.fileType == 1) ? 100 : 200);

        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

        /* -------------------- Device-side memory -------------------- */
        CUDA_CHECK(cudaMalloc(&d_pRawSignal, rawSignalLen * sizeof(short)));
        CUDA_CHECK(cudaMalloc(&d_caCode, codeLen * sizeof(signed char)));
        /* Upload the first code unconditionally. This also covers GLONASS
           frequency channel K = 0, which is a valid cache key. */
        CUDA_CHECK(cudaMemcpy(d_caCode, caCode, codeLen * sizeof(signed char),
            cudaMemcpyHostToDevice));
        PRN = inPRN;
        CUDA_CHECK(cudaMalloc(&d_BasebandSignal, (blksize + 100) * sizeof(GPU_Complex)));

        // Grid size for carrier wipeoff kernels
        blockPerGrid = (blksize + 100 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        /* ----------------- Correlation values ----------------
         * Host mapped page-locked memory is used so the GPU can
         * write correlation results directly to a host-visible buffer.
         *------------------------------------------------------ */
        CUDA_CHECK(cudaHostAlloc(&h_corrValues, CORR_NUMBER * sizeof(std::complex<float>), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_corrValues, h_corrValues, 0));

        initialized = 1;
        printf("   CUDA initialized ...\n");
        printf("   Channel-serial tracking with the GPU-accelerated correlator developed by Yafeng Li.\n");
    } // initialization

    settings.earlyLateSpc = earlyLateSpc;

    /* C ABI arguments are already unpacked pointers and scalars. Compute only
       the number of int16 entries transferred for this call. */
    rawSignalLen = (size_t)rawSignalLenIn;
    blksize = (settings.fileType == 1) ? (int)rawSignalLen : (int)(rawSignalLen / 2);

    /* --------------- Initialize output array ---------------
     * corrValues is a static six-element host array. Every successful call
     * overwrites it, and the Python wrapper copies it before the next call.
     *---------------------------------------------------------- */

    /* -------------------- Copy PRN codes ------------------
     * In channel-serial tracking, only one PRN is processed per call.
     * Reuse the device-side code table until the tracked PRN changes.
     *-------------------------------------------------------- */
    if (PRN != inPRN) {
        CUDA_CHECK(cudaMemcpy(d_caCode, caCode, codeLen * sizeof(signed char), cudaMemcpyHostToDevice));
        PRN = inPRN;
    }

    /* ---------- Copy input IF signals to device --------------- */
    CUDA_CHECK(cudaMemcpy(d_pRawSignal, rawSignal, rawSignalLen * sizeof(short), cudaMemcpyHostToDevice));

    /* ------------ CUDA correlator implementation ------------
     * 1) carrier wipeoff to generate complex baseband samples
     * 2) Early / Prompt / Late correlation with the local code
     *--------------------------------------------------------- */
    if (settings.fileType == 1) {
        mixCarrReal<<<blockPerGrid, THREADS_PER_BLOCK>>>(
            d_pRawSignal, d_BasebandSignal, blksize, (float)remCarrPhase, (float)carrPhaseStep);
    } else {
        mixCarrComplex<<<blockPerGrid, THREADS_PER_BLOCK>>>(
            d_pRawSignal, d_BasebandSignal, blksize, (float)remCarrPhase, (float)carrPhaseStep);
    }
    CUDA_KERNEL_CHECK();

    float initCodePhase = 1.0f - (float)settings.earlyLateSpc + (float)remCodePhase;
    correlator<<<CORR_NUMBER, THREADS_PER_BLOCK>>>(
        d_corrValues, d_BasebandSignal, d_caCode, (float)settings.earlyLateSpc,
        codeLen, (float)codePhaseStep, initCodePhase, blksize);
    CUDA_KERNEL_CHECK();

    /* ------------ Retrieve results from device ----------------- */
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int ind = 0; ind < CORR_NUMBER; ++ind) {
        corrValues[ind * 2] = (double)h_corrValues[ind].real();
        corrValues[ind * 2 + 1] = (double)h_corrValues[ind].imag();
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
__global__ void mixCarrReal(const short* d_pRawSignal, GPU_Complex* d_BasebandSignal, int blksize, float remCarrPhase, float carrPhaseStep)
{
    // CUDA version of floating point NCO and vector dot product integrated

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < blksize; index += blockDim.x * gridDim.x) {
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
 * Compute Early / Prompt / Late correlations.
 * blockIdx.x selects the correlator branch:
 *   vec = 0 -> Early
 *   vec = 1 -> Prompt
 *   vec = 2 -> Late
 *
 * Each thread accumulates a strided partial sum, then a shared-memory tree reduction
 * combines them to obtain one complex correlation value per branch.
 */
__global__ void correlator(GPU_Complex* d_corrValues, GPU_Complex* d_BasebandSignal, const signed char* __restrict__ d_caCodeTable,
    float earlyLateSpc, int codeLen, float codePhaseStep, float initCodePhase, int blksize)
{
    //Accumulators cache
    __shared__ GPU_Complex accumResult[ACCUM_N];

    for (int vec = blockIdx.x; vec < CORR_NUMBER; vec += gridDim.x)
    {
        float initPhase = initCodePhase + earlyLateSpc * vec;

        for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
        {
            GPU_Complex sumIQ(0.0f, 0.0f);

            //float code_phase;
            for (int pos = iAccum; pos < blksize; pos += ACCUM_N) {
                // 1.resample local code for the current shift
                float chipIndex = fmodf(codePhaseStep * __int2float_rd(pos) + initPhase, codeLen);

                // 2.correlate
                sumIQ.multiply_acc(d_BasebandSignal[pos], (float)__ldg(&d_caCodeTable[__float2int_rd(chipIndex)]));
            }
            accumResult[iAccum] = sumIQ;
        }

        // Tree reduction in shared memory. ACCUM_N must be a power of two.
        for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                accumResult[iAccum] += accumResult[stride + iAccum];
            }
        }

        if (threadIdx.x == 0) {
            d_corrValues[vec] = accumResult[0];
        }
    }
}

/* checkInputs ----------------------------------------------------------------
 * Validate the fixed interface values once, before persistent buffers are
 * allocated. Later calls reuse the initialized configuration.
 * Args   : int fileType              I   1 - real; 2 - interleaved complex
 *          int rawSignalLenIn        I   number of int16 elements in rawSignal
 *          const short* rawSignal    I   input int16 IF signal
 *          const signed char* caCode I   guarded int8 local code
 *          int codeLen               I   guarded-code element count
 * Return : 1 for valid input; 0 otherwise
 *------------------------------------------------------------------------------*/
static int checkInputs(int fileType, int rawSignalLenIn, const short* rawSignal,
    const signed char* caCode, int codeLen)
{
    if (rawSignal == NULL || caCode == NULL || rawSignalLenIn <= 0 ||
        codeLen <= 0 || (fileType != 1 && fileType != 2)) {
        printf("   Invalid input arguments for corrGPUSerialBPSK.\n");
        return 0;
    }
    return 1;
}

/* cleanup --------------------------------------------------------------------
 * Release persistent device and host-side resources owned by this DLL. The CUDA primary
 * context is not reset or destroyed because it may also be used by CuPy.
 * It is safe to call repeatedly and a later corrEngine() call can initialize
 * a new tracking run.
 * Return : None
 *------------------------------------------------------------------------------*/
void cleanup(void)
{
    printf("   CUDA is terminating, destroying allocated memory ...\n");

    /* --------- Free device-side memory --------- */
    if (d_pRawSignal != NULL) {
        (void)cudaFree(d_pRawSignal);
        d_pRawSignal = NULL;
    }

    if (d_caCode != NULL) {
        (void)cudaFree(d_caCode);
        d_caCode = NULL;
    }

    if (d_BasebandSignal != NULL) {
        (void)cudaFree(d_BasebandSignal);
        d_BasebandSignal = NULL;
    }

    /* --------- Free host-side memory --------- */
    if (h_corrValues != NULL) {
        (void)cudaFreeHost(h_corrValues);
        h_corrValues = NULL;
        d_corrValues = NULL;
    }

    /* --------- Reset persistent state --------- */
    initialized = 0;
}
