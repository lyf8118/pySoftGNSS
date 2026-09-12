/*================================================================================
 * Filename: corrSIMDSerialBPSK.cpp
 * Description: AVX2 SIMD single-channel tracking correlator exposed through
 *              a small C ABI for Python ctypes.
 *
 * Authors: Yafeng Li (School of Automation, Beijing Information Science and Technology University)
 * Time: Aug, 1, 2026
 *
 * The core processing stages are intentionally kept aligned with the
 * channel-parallel SIMD correlator:
 *   1) mixCarrReal() / mixCarrComplex()
 *   2) localCodeGen()
 *   3) correlator()
 *
 * This file processes one channel and one code period per call. The first
 * corrEngine() call allocates reusable work buffers; corrEngineFree() releases
 * them after channel-serial tracking has finished.
 *==============================================================================*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>

typedef size_t mwSize;

#define DLL_EXPORT extern "C" __declspec(dllexport)

/* Settings cached from the first call. fileType and rShiftBits remain fixed
 * during one tracking run; earlyLateSpc is deliberately supplied on every
 * corrEngine() call. */
typedef struct
{
	int fileType;
	int rShiftBits;
	double earlyLateSpc; /* half of the early - late code correlation spacing(chips) */
} Settings;

/* Function declaration ---------------------------------------------------------
 * mixCarrReal()        : mix local carrier with real IF data
 * mixCarrComplex()     : mix local carrier with complex IF data
 * localCodeGen()       : generate early/prompt/late local code sequences
 * correlator()         : compute IE/QE/IP/QP/IL/QL correlation outputs
 * cleanup()            : release persistent host buffers
 * checkedMalloc()      : allocate non-aligned host memory
 * checkedAlignedMalloc(): allocate 32-byte aligned host memory for SIMD buffers
 * gatherCode16()       : gather 16 local-code samples and repack them to int16 order
 *------------------------------------------------------------------------------*/
void mixCarrReal(const short*, double, double, mwSize, short*, short*);
void mixCarrComplex(const short*, const short*, double, double, mwSize, short*, short*);
void localCodeGen(const int*, mwSize, double, double, double, mwSize, short**);
void correlator(short*, short*, short**, mwSize, double*);
void cleanup(void);
static int checkInputs(int fileType, int blksize, const short* rawSignal,
	const int* caCode, int codeLen);
static void* checkedMalloc(size_t nbytes, const char* name);
static void* checkedAlignedMalloc(size_t nbytes, const char* name);
static __m256i gatherCode16(const int* caCode, __m256i Index_reg1, __m256i Index_reg2);

/* Constants -------------------------------------------------------------------- */
#define PI          3.1415926535897932  // pi 
#define DPI         (2.0*PI)            // 2*pi 
#define CSCALE      (1.0/16.0)          // carrier lookup table scale (LSB) 

/* Persistent buffers -----------------------------------------------------------
 * initialized     : initialization flag for persistent buffers
 * iBasebandSignal : in-phase baseband samples after carrier wipeoff
 * qBasebandSignal : quadrature baseband samples after carrier wipeoff
 * rawSignalI/Q    : split I/Q branches for complex input data
 * localCode       : local code buffers for early, prompt and late replicas
 * settings        : file type and input right shift cached on the first call
 *------------------------------------------------------------------------------*/
static int initialized = 0;
static short* iBasebandSignal = NULL;
static short* qBasebandSignal = NULL;
static short* rawSignalI = NULL;
static short* rawSignalQ = NULL;
static short** localCode = NULL;
static Settings settings;

/* Python ctypes entry point ---------------------------------------------------
 * Args   : fileType       I   1 - real samples; 2 - interleaved I/Q samples
 *          blksize        I   number of int16 elements in rawSignal
 *          rShiftBits     I   input right shift used by the SIMD mixer
 *          earlyLateSpc   I   half early-late spacing (chips)
 *          rawSignal      I   int16 IF samples
 *          caCode         I   int32 guarded code [last, one period, first]
 *          codeLen        I   number of elements in caCode
 *          remCarrPhase   I   residual carrier phase (rad)
 *          carrPhaseStep  I   carrier phase increment (rad/sample)
 *          remCodePhase   I   residual code phase (chips)
 *          codePhaseStep  I   code phase increment (chips/sample)
 * Return : pointer to static [IE QE IP QP IL QL], or NULL on invalid input or
 *          first-call allocation failure. The pointer is overwritten by the
 *          next corrEngine() call.
 *------------------------------------------------------------------------------*/
DLL_EXPORT double* corrEngine(int fileType, int blksize, int rShiftBits,
	double earlyLateSpc, short* rawSignal, int* caCode, int codeLen,
	double remCarrPhase, double carrPhaseStep, double remCodePhase,
	double codePhaseStep)
{
	static double corrValues[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

	/* Allocate once from the first logical block size. Tracking guarantees that
	   subsequent logical blocks do not exceed this size + 100 samples. */
	if (!initialized) {
		/* ----------------- Validate and cache the serial settings once -----------------
		   This serial wrapper assumes the C ABI settings stay unchanged after the first
		   call. If settings or input types change later, call corrEngineFree() and
		   reinitialize so the cached framework state matches the new configuration. */
		// gateway check
		if (!checkInputs(fileType, blksize, rawSignal, caCode, codeLen)) {
			return NULL;
		}
		
		settings.fileType = fileType;
		settings.rShiftBits = rShiftBits;
		
		size_t rawSignalLen = (size_t)blksize;
		if (settings.fileType == 2) {
			rawSignalLen = rawSignalLen / 2;
		}
		rawSignalLen = rawSignalLen + 100;
		iBasebandSignal = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "iBasebandSignal");
		qBasebandSignal = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "qBasebandSignal");

		localCode = (short**)checkedMalloc(sizeof(short*) * 3, "localCode");
		if (localCode != NULL) {
			for (int i = 0; i < 3; ++i) {
				localCode[i] = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "localCode[i]");
			}
		}

		if (settings.fileType == 2) {
			rawSignalI = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "rawSignalI");
			rawSignalQ = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "rawSignalQ");
		}

		if (iBasebandSignal == NULL || qBasebandSignal == NULL ||
			localCode == NULL || localCode[0] == NULL ||
			localCode[1] == NULL || localCode[2] == NULL ||
			(fileType == 2 && (rawSignalI == NULL || rawSignalQ == NULL))) {
			cleanup();
			return NULL;
		}

		initialized = 1;
		printf("   SIMD DLL initialized ...\n");
		printf("   Channel-serial tracking with the SIMD-accelerated correlator developed by Yafeng Li.\n");
	}

	/* fileType and rShiftBits are fixed for the lifetime of the initialized
	   engine. earlyLateSpc and all phase parameters remain per-call inputs. */
	settings.earlyLateSpc = earlyLateSpc;

	if (settings.fileType == 2) {
		blksize = blksize / 2;
		/* Split the interleaved serial rawSignal block into separate I/Q buffers so the
		   complex SIMD mixer can reuse the exact same implementation as corrSIMDParallelBPSK.cpp. */
		for (int k = 0; k < blksize; ++k) {
			rawSignalI[k] = rawSignal[2 * k];
			rawSignalQ[k] = rawSignal[2 * k + 1];
		}
	}

	/* --------------------- Initialize output and run the three-stage core ---------------------
	   The output layout follows trkChannelsSerial.m directly:
	     [IE QE IP QP IL QL]
	   and the processing stages are intentionally unchanged from the shared SIMD core. */
	if (settings.fileType == 1) {
		mixCarrReal(rawSignal, carrPhaseStep, remCarrPhase, (mwSize)blksize, iBasebandSignal, qBasebandSignal);
	} else {
		mixCarrComplex(rawSignalI, rawSignalQ, carrPhaseStep, remCarrPhase, (mwSize)blksize, iBasebandSignal, qBasebandSignal);
	}
	localCodeGen(caCode, (mwSize)codeLen, remCodePhase, codePhaseStep, settings.earlyLateSpc, (mwSize)blksize, localCode);
	correlator(iBasebandSignal, qBasebandSignal, localCode, (mwSize)blksize, corrValues);
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

/* Mix local carrier for real data -----------------------------------------------
 * Mix local carrier to input signal
 * Args   : short   *rawSignal         I   Input IF signal
 *          double carrPhaseStep       I   carrier sampling interval (cycle: rad*t)
 *          double remCarrPhase        I   initial carrier phase (rad)
 *          mwSize blksize             I   number of samples to be generated
 *          short  *iBasebandSignal    O   I component of input signal with carrier wiped off 
 *          short  *qBasebandSignal    O   Q component of input signal with carrier wiped off
 * Return : None
 *------------------------------------------------------------------------------*/
void mixCarrReal(const short *rawSignal, double carrPhaseStep, double remCarrPhase, mwSize blksize,
	         short *iBasebandSignal, short *qBasebandSignal)
{
	
	/* ---------- Inphase and quadrature carrier lookup table --------- */
	static char cost[16] = { 0 }, sint[16] = { 0 };  
	//Carrier lookup table for exp(-j*carrPhase)
	if (!cost[0]) {
        for (int i = 0; i < 16; ++i) {
            cost[i] = (char)floor(cos(PI / 8.0 * i) / CSCALE + 0.5);
            sint[i] = (char)floor(-sin(PI / 8.0 * i) / CSCALE + 0.5);
		}
	}

	/* Convert radians to a 16-state phase accumulator used by the LUT.
	   One integer step corresponds to pi/8, so phase indexes are simply
	   obtained by rounding the scaled phase and masking with 0x0F. */
    remCarrPhase = remCarrPhase * 16.0 / DPI;
	/* Carrier phase step */
    const double phaseStep = carrPhaseStep * 16.0 / DPI;

	__m256i Index_mm256, Index_reg1, Index_reg2, cos_reg_mm256, sin_reg_mm256, IF_reg, iBaseband_reg, qBaseband_reg;
	__m256 carrPhase_reg1, carrPhase_reg2;
	__m128i Index_reg, cos_reg_mm128, sin_reg_mm128;
	/* The seemingly odd order compensates for the lane-local packing order of
	   _mm256_packs_epi32. After the later pack/shuffle/permute sequence, the
	   resulting 16 carrier samples still map to chronological sample order
	   0, 1, 2, ..., 15. */
	__m256 stepNumber_reg = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 8.0f, 9.0f, 10.0f, 11.0f);
	const __m256 remCarrPhase_reg = _mm256_set1_ps((float)remCarrPhase);
	const __m256 carrPhaseStep_reg = _mm256_set1_ps((float)phaseStep);
	const __m256i mask4 = _mm256_set1_epi32(15);
	const __m256 sixteenStep = _mm256_set1_ps(16.0f);
	const __m256 fourPhaseStep = _mm256_set1_ps((float)(phaseStep * 4.0));
	
	const __m128i local_cos = _mm_loadu_si128((__m128i *)cost);   
	const __m128i local_sin = _mm_loadu_si128((__m128i *)sint);

    const short *IFData = rawSignal;
	for ( ; IFData <= rawSignal + blksize - 16; IFData+=16,iBasebandSignal+=16,qBasebandSignal+=16)
	{
		/* ------------------ Inphase and quadrature carriers generation ------------------ */
		/* Generate the carrier phase of 16 consecutive samples. Each AVX register
		   holds 8 phase values; together carrPhase_reg1/carrPhase_reg2 cover the
		   16-sample block that will later be packed back into time order. */
		carrPhase_reg1 = _mm256_fmadd_ps(carrPhaseStep_reg, stepNumber_reg, remCarrPhase_reg);
		carrPhase_reg2 = _mm256_add_ps(carrPhase_reg1, fourPhaseStep);
		
		/* Convert phase to LUT indexes. cvtps rounds to nearest integer, which
		   reduces carrier quantization error compared with truncation. */
		Index_reg1 = _mm256_cvtps_epi32(carrPhase_reg1);   // using _mm256_cvttps_epi32 will result in larger error    
		Index_reg1 = _mm256_and_si256(Index_reg1, mask4);   
		Index_reg2 = _mm256_cvtps_epi32(carrPhase_reg2); 
		Index_reg2 = _mm256_and_si256(Index_reg2, mask4);
		
		/* Pack 8+8 int32 indexes to 16 int16 indexes. The chosen step-number order
		   makes the low/high 128-bit halves already correspond to samples 0..7
		   and 8..15 respectively. */
		Index_mm256 = _mm256_packs_epi32(Index_reg1, Index_reg2);
		
		/* The LUT index range is already 0..15, so each packed int16 index has a
		   zero high byte. Extract the two 128-bit halves and pack them directly
		   to 16 contiguous bytes for the byte-wise LUT shuffle. */
		Index_reg = _mm_packus_epi16(_mm256_castsi256_si128(Index_mm256),
			_mm256_extracti128_si256(Index_mm256, 1));

		/* Look up 16 quantized carrier samples from the 16-entry cosine/sine LUTs.
		   Each byte in Index_reg selects one byte from local_cos/local_sin. */
		cos_reg_mm128 = _mm_shuffle_epi8(local_cos, Index_reg);
		sin_reg_mm128 = _mm_shuffle_epi8(local_sin, Index_reg);

		/* Expand carrier samples to int16 so they can be multiplied directly with
		   the int16 IF samples. Carrier amplitude is quantized by CSCALE = 1/16. */
		cos_reg_mm256 = _mm256_cvtepi8_epi16(cos_reg_mm128);
		sin_reg_mm256 = _mm256_cvtepi8_epi16(sin_reg_mm128);

		/* ----------------- Carrier wiping off from the input IF signals ----------------- */
		/* Load 16 IF samples. The persistent buffers were 32-byte aligned, so the
		   later stores can use aligned writes even though the input may be unaligned. */
		IF_reg = _mm256_loadu_si256((__m256i *)(IFData));
        /* Right-shift IF samples when ADC valid bits occupy the high bits. */
        if (settings.rShiftBits != 0) {
            IF_reg = _mm256_srai_epi16(IF_reg, settings.rShiftBits);
        }
		
		/* Real IF data multiplied by exp(-j*phase):
		     I = x * cos(phase)
		     Q = x * (-sin(phase))
		   The minus sign is already baked into sint[] in mixCarrReal(). */
		iBaseband_reg = _mm256_mullo_epi16(IF_reg, cos_reg_mm256);
		qBaseband_reg = _mm256_mullo_epi16(IF_reg, sin_reg_mm256);

		/* Store the de-rotated baseband block. localCodeGen() and correlator()
		   will reuse these buffers immediately after this stage. */
		_mm256_store_si256((__m256i*)iBasebandSignal, iBaseband_reg);
		_mm256_store_si256((__m256i*)qBasebandSignal, qBaseband_reg);

		/* Advance the synthetic sample index by 16 samples for the next SIMD block. */
		stepNumber_reg = _mm256_add_ps(stepNumber_reg, sixteenStep);
	}	

	/* ----------------- For sampling points outside the SIMD iteration  ----------------- */
	/* Convert the SIMD-loop progress back into the scalar phase accumulator so the
	   tail loop continues from exactly the same quantized carrier state. */
	remCarrPhase = (blksize / 16) * 16 * phaseStep + remCarrPhase;
	
    for (; IFData < rawSignal + blksize; ++IFData, ++iBasebandSignal, ++qBasebandSignal) {
        short sample = IFData[0];
        if (settings.rShiftBits != 0) {
            sample = (short)(sample >> settings.rShiftBits);
        }
		/* Scalar fallback uses the same LUT and scaling as the SIMD path, so the
		   last 0..15 samples remain bit-consistent with the vectorized section. */
		int n = ((int)remCarrPhase) % 16;
        iBasebandSignal[0] = (short)(cost[n] * sample);
        qBasebandSignal[0] = (short)(sint[n] * sample);
        remCarrPhase += phaseStep;
	}
}

/* Mix local carrier for complex data --------------------------------------------
 * Mix local carrier to input signal
 * Args   : short   *rawSignalI        I   Real part of the input complex IF signal
 *          short   *rawSignalQ        I   Imaginary part of the input complex IF signal
 *          double carrPhaseStep       I   carrier sampling interval (cycle: rad*t)
 *          double remCarrPhase        I   Initial carrier phase (rad)
 *          mwSize blksize             I   Number of samples to be generated
 *          short  *iBasebandSignal    O   I component of input signal with carrier wiped off
 *          short  *qBasebandSignal    O   Q component of input signal with carrier wiped off
 * Return : None
 *------------------------------------------------------------------------------*/
void mixCarrComplex(const short* rawSignalI, const short* rawSignalQ, double carrPhaseStep, double remCarrPhase, mwSize blksize,
	short* iBasebandSignal, short* qBasebandSignal)
{
	/* Inphase and quadrature carrier lookup table */
	static char cost[16] = { 0 }, sint[16] = { 0 };

	//Carrier lookup table for exp(-j*carrPhase)
	if (!cost[0]) 	{
		for (int i = 0; i < 16; ++i) {
			cost[i] = (char)floor(cos(PI / 8.0 * i) / CSCALE + 0.5);
			sint[i] = (char)floor(sin(PI / 8.0 * i) / CSCALE + 0.5);
		}
	}

	/* Complex carrier wipeoff uses the same 16-state LUT idea as the real-data
	   version. Here sint[] is positive, and the signs are introduced later by
	   the complex multiply formulas for exp(-j*phase). */
    remCarrPhase = remCarrPhase * 16.0 / DPI;
	/* Carrier phase step */
    const double phaseStep = carrPhaseStep * 16.0 / DPI;

	__m256i Index_mm256, Index_reg1, Index_reg2, cos_reg_mm256, sin_reg_mm256, IF_regI, IF_regQ, \
		sin_regI, cos_regI, sin_regQ, cos_regQ, iBaseband_reg, qBaseband_reg;
	__m256 carrPhase_reg1, carrPhase_reg2;
	__m128i Index_reg, cos_reg_mm128, sin_reg_mm128;
	
	/* Same index-order trick as mixCarrReal(): setr_ps makes the lane contents
	   explicit as [0 1 2 3 | 8 9 10 11], which still compacts to samples
	   0, 1, 2, ..., 15 after the later pack-and-compact path. */
	__m256 stepNumber_reg = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 8.0f, 9.0f, 10.0f, 11.0f);
	const __m256 remCarrPhase_reg = _mm256_set1_ps((float)remCarrPhase);
	const __m256 carrPhaseStep_reg = _mm256_set1_ps((float)phaseStep);
	const __m256i mask4 = _mm256_set1_epi32(15);
	const __m256 sixteenStep = _mm256_set1_ps(16.0f);
	const __m256 fourPhaseStep = _mm256_set1_ps((float)(phaseStep * 4.0));
	const __m128i local_cos = _mm_loadu_si128((__m128i*)cost);
	const __m128i local_sin = _mm_loadu_si128((__m128i*)sint);

    const short *IFDataI = rawSignalI;
    const short *IFDataQ = rawSignalQ;
    for (; IFDataI <= rawSignalI + blksize - 16; IFDataI += 16, IFDataQ += 16, iBasebandSignal += 16, qBasebandSignal += 16) 
	{
		/* ------------------ Inphase and quadrature carriers generation ------------------ */
		/* Build the carrier phase for 16 adjacent complex samples. */
		carrPhase_reg1 = _mm256_fmadd_ps(carrPhaseStep_reg, stepNumber_reg, remCarrPhase_reg);
		carrPhase_reg2 = _mm256_add_ps(carrPhase_reg1, fourPhaseStep);

		/* Convert those phases into 0..15 LUT indexes. */
		Index_reg1 = _mm256_cvtps_epi32(carrPhase_reg1);   // using _mm256_cvttps_epi32 will result in larger error    
		Index_reg1 = _mm256_and_si256(Index_reg1, mask4);
		Index_reg2 = _mm256_cvtps_epi32(carrPhase_reg2);
		Index_reg2 = _mm256_and_si256(Index_reg2, mask4);

		/* Pack and compact the LUT indexes so they can feed a byte-wise pshufb lookup. */
		Index_mm256 = _mm256_packs_epi32(Index_reg1, Index_reg2);

		/* The LUT index range is already 0..15, so each packed int16 index has a
		   zero high byte. Extract the two 128-bit halves and pack them directly
		   to 16 contiguous bytes for the byte-wise LUT shuffle. */
		Index_reg = _mm_packus_epi16(_mm256_castsi256_si128(Index_mm256),
			_mm256_extracti128_si256(Index_mm256, 1));

		/* Fetch the 16 cosine and sine coefficients from the quantized LUTs. */
		cos_reg_mm128 = _mm_shuffle_epi8(local_cos, Index_reg);
		sin_reg_mm128 = _mm_shuffle_epi8(local_sin, Index_reg);

		/* Expand LUT output to int16 for the complex multiply below. */
		cos_reg_mm256 = _mm256_cvtepi8_epi16(cos_reg_mm128);
		sin_reg_mm256 = _mm256_cvtepi8_epi16(sin_reg_mm128);

		/* ----------------- Carrier wiping off from the input IF signals ----------------- */
		/* Load 16 complex samples split as I and Q arrays. */
		IF_regI = _mm256_loadu_si256((__m256i*)IFDataI);
		IF_regQ = _mm256_loadu_si256((__m256i*)IFDataQ);
		
        if (settings.rShiftBits != 0) {
            IF_regI = _mm256_srai_epi16(IF_regI, settings.rShiftBits);
            IF_regQ = _mm256_srai_epi16(IF_regQ, settings.rShiftBits);
		}

		/* Compute the four products needed for:
		     (I + jQ) * (cos - j sin)
		   and then combine them into the real/imag parts of the baseband signal. */
		sin_regI = _mm256_mullo_epi16(IF_regI, sin_reg_mm256);
		cos_regI = _mm256_mullo_epi16(IF_regI, cos_reg_mm256);
		sin_regQ = _mm256_mullo_epi16(IF_regQ, sin_reg_mm256);
		cos_regQ = _mm256_mullo_epi16(IF_regQ, cos_reg_mm256);

		/* Combine the four components to real and imag part of the complex baseband */
		/* iBasebandSignal = real(exp(-j*carrPhase).*rawSignal);
		   qBasebandSignal = imag(exp(-j*carrPhase).*rawSignal); */
		iBaseband_reg = _mm256_add_epi16(cos_regI, sin_regQ);
		qBaseband_reg = _mm256_sub_epi16(cos_regQ, sin_regI);
		

		/* Save the de-rotated complex baseband samples for the code correlator. */
		_mm256_store_si256((__m256i*)iBasebandSignal, iBaseband_reg);
		_mm256_store_si256((__m256i*)qBasebandSignal, qBaseband_reg);

		stepNumber_reg = _mm256_add_ps(stepNumber_reg, sixteenStep);
	}

	/* ----------------- For sampling points outside the SIMD iteration  ----------------- */
	/* Reconstruct the scalar phase state that follows the vectorized section. */
	remCarrPhase = (blksize / 16) * 16 * phaseStep + remCarrPhase;

	for (; IFDataI < rawSignalI + blksize; ++IFDataI, ++IFDataQ, ++iBasebandSignal, ++qBasebandSignal)
	{
        short sampleI = IFDataI[0];
        short sampleQ = IFDataQ[0];
        if (settings.rShiftBits != 0) {
            sampleI = (short)(sampleI >> settings.rShiftBits);
            sampleQ = (short)(sampleQ >> settings.rShiftBits);
        }
		/* Scalar fallback uses the same complex multiply as the SIMD path:
		     Ibb =  I*cos + Q*sin
		     Qbb =  Q*cos - I*sin */
		int n = ((int)remCarrPhase) % 16;
        iBasebandSignal[0] = (short)(cost[n] * sampleI + sint[n] * sampleQ);
        qBasebandSignal[0] = (short)(cost[n] * sampleQ - sint[n] * sampleI);
		remCarrPhase += phaseStep;
	}
}

 /* Generate local code ----------------------------------------------------------
 * Generate local code sequences
 * Args   : int     *caCode          I   int32 GPS L1 PRN code sequence 
 *          mwSize codeLen           I   chip number of the PRN code
 *          double remCodePhase      I   initial code phase (chip)
 *          double codePhaseStep     I   code sampling interval (chip)
 *          double earlyLateSpc      I   half of the early-late code correlation spacing (chips)
 *          int    blksize           I   number of samples to be generated 
 *          short  *localCode        O   local code replica outputs
 * return : None 
 *------------------------------------------------------------------------------*/
void localCodeGen(const int *caCode, mwSize codeLen, double remCodePhase, double codePhaseStep,
	              double earlyLateSpc, mwSize blksize, short **localCode)
{
	/* Initial code phase of early code: 1.0 is due to the first code
	chip added to caCode for early replica generation */
	remCodePhase = remCodePhase - earlyLateSpc + 1.0;
	if (remCodePhase >= codeLen)
		remCodePhase -= floor(remCodePhase / codeLen)*codeLen;
	
	/* caCode is laid out as [lastChip, code(1..1023), firstChip]. Shifting the
	   starting phase by -earlyLateSpc + 1.0 means the early replica can safely
	   address one chip before the nominal prompt position, and the late replica
	   can safely walk one chip beyond the end without needing modulo per sample. */
	short *eCode = localCode[0], *pCode = localCode[1], * lCode = localCode[2];

	__m256i Index_reg1, Index_reg2, EPL_reg;
	__m256 codePhase_reg1, codePhase_reg2;
	/* Here the natural 0..7 order is fine because code indexes are gathered as
	   two 8-lane AVX2 vectors and packed back to 16 x int16 code chips. */
	__m256 stepNumber_reg = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
	const __m256 remCodePhase_reg = _mm256_set1_ps((float)remCodePhase);
	const __m256 codePhaseStep_reg = _mm256_set1_ps((float)codePhaseStep);
	const __m256 earlyLateSpc_reg = _mm256_set1_ps((float)earlyLateSpc);
	const __m256 sixteenStep = _mm256_set1_ps(16.0f);
	const __m256 eightCodePhase = _mm256_set1_ps((float)(codePhaseStep * 8.0));

	for ( ; eCode <= localCode[0] + blksize - 16; eCode += 16, pCode += 16, lCode += 16)
	{
		/* ----------------------- Early code generation ----------------------- */
		/* Generate the code phase of 16 successive samples. Because the phase is
		   always non-negative in this routine, truncation toward zero is equivalent
		   to floor(), matching the MATLAB integer indexing behavior. */
		codePhase_reg1 = _mm256_fmadd_ps(codePhaseStep_reg, stepNumber_reg, remCodePhase_reg);
		codePhase_reg2 = _mm256_add_ps(codePhase_reg1, eightCodePhase);
		Index_reg1 = _mm256_cvttps_epi32(codePhase_reg1);
		Index_reg2 = _mm256_cvttps_epi32(codePhase_reg2);
		/* Gather 16 local-code chips directly from the int32 code table,
		   then pack them back to int16 so correlator() can keep using madd_epi16. */
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		_mm256_store_si256((__m256i*)eCode, EPL_reg);

		/* ----------------------- Prompt code generation ----------------------- */
		/* Prompt is just Early shifted by +earlyLateSpc chips. */
		codePhase_reg1 = _mm256_add_ps(codePhase_reg1,earlyLateSpc_reg);
		codePhase_reg2 = _mm256_add_ps(codePhase_reg2, earlyLateSpc_reg);
		Index_reg1 = _mm256_cvttps_epi32(codePhase_reg1);
		Index_reg2 = _mm256_cvttps_epi32(codePhase_reg2);
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		_mm256_store_si256((__m256i*)pCode, EPL_reg);

		/* ----------------------- Late code generation ----------------------- */
		/* Late is Prompt shifted once more by +earlyLateSpc, i.e. Early + 2*spacing. */
		codePhase_reg1 = _mm256_add_ps(codePhase_reg1, earlyLateSpc_reg);
		codePhase_reg2 = _mm256_add_ps(codePhase_reg2, earlyLateSpc_reg);
		Index_reg1 = _mm256_cvttps_epi32(codePhase_reg1);
		Index_reg2 = _mm256_cvttps_epi32(codePhase_reg2);
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		_mm256_store_si256((__m256i*)lCode, EPL_reg);

		/* Move to the next batch of 16 code samples. */
		stepNumber_reg = _mm256_add_ps(stepNumber_reg, sixteenStep);
	}

	/* ------------ For sampling points outside the SIMD iteration  ------------ */
	/* Recreate the code phase at the first leftover sample after the SIMD loop. */
	double tempCodePhase = (blksize / 16) * 16 * codePhaseStep + remCodePhase;
	double twoEarlyLateSpc = earlyLateSpc * 2;
	for (; eCode < localCode[0] + blksize; ++eCode, ++pCode, ++lCode)
	{
		/* Scalar tail keeps the same Early/Prompt/Late geometry as the SIMD path. */
		eCode[0] = caCode[(int)tempCodePhase];
		pCode[0] = caCode[(int)(tempCodePhase + earlyLateSpc)];
		lCode[0] = caCode[(int)(tempCodePhase + twoEarlyLateSpc)];
		tempCodePhase += codePhaseStep;
	}
}

/* Correlating function ------------------------------------------------------------
 * Mix local carrier to input signal
 * Args :   short  *iBasebandSignal    I   I component of input signal with carrier wiped off
 *          short  *qBasebandSignal    I   Q component of input signal with carrier wiped off
 *          short  **localCode         I   local code sequences
 *          mwSize blksize             I   number of samples to be generated
 *          double *corrValues        O   outputs of correlator values arranged as [IE QE IP QP IL QL] 
 * Return : None
 *------------------------------------------------------------------------------*/
void correlator(short *iBasebandSignal, short *qBasebandSignal, short **localCode, mwSize blksize, double *corrValues)
{
	int I_E_sum[8], I_P_sum[8], I_L_sum[8], Q_E_sum[8], Q_P_sum[8], Q_L_sum[8];
	long long I_E = 0, I_P = 0, I_L = 0, Q_E = 0, Q_P = 0, Q_L = 0;
	
	const short* pI = iBasebandSignal;
	const short* pQ = qBasebandSignal;
	short* eCode = localCode[0];
	short* pCode = localCode[1];
	short* lCode = localCode[2];

	/* Each local code sample is stored as int16 +/-1. _mm256_madd_epi16 performs:
	     [a0*b0 + a1*b1, a2*b2 + a3*b3, ...]
	   so one AVX instruction produces eight int32 partial sums from 16 samples. */
	__m256i I_reg, Q_reg, I_E_reg, I_P_reg, I_L_reg, Q_E_reg, Q_P_reg, Q_L_reg, EPL_reg, mul_reg;
	I_E_reg = _mm256_setzero_si256();
	I_P_reg = _mm256_setzero_si256();
	I_L_reg = _mm256_setzero_si256();
	Q_E_reg = _mm256_setzero_si256();
	Q_P_reg = _mm256_setzero_si256();
	Q_L_reg = _mm256_setzero_si256();
	
	for (; eCode <= localCode[0] + blksize - 16; eCode += 16, pCode += 16, lCode += 16, pI += 16, pQ += 16){
		
		/* --------------- Load IF signal with carrier wiped off --------------- */
		/* These loads are aligned because iBasebandSignal/qBasebandSignal/localCode
		   were allocated with 32-byte alignment and advanced in 16-sample steps. */
		I_reg = _mm256_load_si256((__m256i *)pI);
		Q_reg = _mm256_load_si256((__m256i *)pQ);

		/* --------------- Compute the early correlation value for data channel --------------- */
		EPL_reg = _mm256_load_si256((__m256i *)eCode);
		/* Multiply 16 samples by 16 code chips and horizontally add adjacent pairs,
		   leaving eight int32 partial sums to accumulate in the AVX register. */
		mul_reg = _mm256_madd_epi16(I_reg, EPL_reg);  //Multiplies signed packed 16-bit integer data elements of two vectors. 
		I_E_reg = _mm256_add_epi32(I_E_reg, mul_reg);
		mul_reg = _mm256_madd_epi16(Q_reg, EPL_reg);
		Q_E_reg = _mm256_add_epi32(Q_E_reg, mul_reg);

		/* --------------- Compute the prompt correlation value for data channel --------------- */
		EPL_reg = _mm256_load_si256((__m256i *)pCode);
		mul_reg = _mm256_madd_epi16(I_reg, EPL_reg);
		I_P_reg = _mm256_add_epi32(I_P_reg, mul_reg);
		mul_reg = _mm256_madd_epi16(Q_reg, EPL_reg);
		Q_P_reg = _mm256_add_epi32(Q_P_reg, mul_reg);

		/* --------------- Compute the late correlation value for data channel --------------- */
		EPL_reg = _mm256_load_si256((__m256i *)lCode);
		mul_reg = _mm256_madd_epi16(I_reg, EPL_reg);
		I_L_reg = _mm256_add_epi32(I_L_reg, mul_reg);
		mul_reg = _mm256_madd_epi16(Q_reg, EPL_reg);
		Q_L_reg = _mm256_add_epi32(Q_L_reg, mul_reg);
	}

	/* --------------- Integration for SIMD register elements ---------------
 * The SIMD partial sums are first stored to scalar arrays and then reduced
 * to 64-bit accumulators to reduce the risk of overflow in long integrations.
 *---------------------------------------------------------------------- */
	_mm256_storeu_si256((__m256i*)I_E_sum, I_E_reg);
	_mm256_storeu_si256((__m256i*)Q_E_sum, Q_E_reg);
	_mm256_storeu_si256((__m256i*)I_P_sum, I_P_reg);
	_mm256_storeu_si256((__m256i*)Q_P_sum, Q_P_reg);
	_mm256_storeu_si256((__m256i*)I_L_sum, I_L_reg);
	_mm256_storeu_si256((__m256i*)Q_L_sum, Q_L_reg);
    
	for (int i = 0; i < 8; ++i) {
		I_E += I_E_sum[i]; Q_E += Q_E_sum[i];
		I_P += I_P_sum[i]; Q_P += Q_P_sum[i];
		I_L += I_L_sum[i]; Q_L += Q_L_sum[i];
	}

	/* ----------------- For sampling points outside the SIMD iteration  ----------------- */
	/* Scalar cleanup directly accumulates the remaining 0..15 samples. */
	for (; eCode < localCode[0] + blksize; ++pI, ++pQ, ++eCode, ++pCode, ++lCode){
		I_E += (long long)pI[0] * (long long)eCode[0];
		Q_E += (long long)pQ[0] * (long long)eCode[0];
		I_P += (long long)pI[0] * (long long)pCode[0];
		Q_P += (long long)pQ[0] * (long long)pCode[0];
		I_L += (long long)pI[0] * (long long)lCode[0];
		Q_L += (long long)pQ[0] * (long long)lCode[0];
	}
	/* Convert the six correlator outputs to double precision and undo the 16x
	   carrier LUT amplitude scaling introduced during carrier wipeoff. */
    corrValues[0] = (double)I_E * CSCALE;
    corrValues[1] = (double)Q_E * CSCALE;
    corrValues[2] = (double)I_P * CSCALE;
    corrValues[3] = (double)Q_P * CSCALE;
    corrValues[4] = (double)I_L * CSCALE;
    corrValues[5] = (double)Q_L * CSCALE;
}



/* checkedMalloc ---------------------------------------------------------------
 * Allocate non-aligned host memory and report allocation failure to corrEngine().
 * Args   : size_t nbytes             I   number of bytes to allocate
 *          const char* name          I   buffer name for error reporting
 * Return : void*                        allocated host pointer
 *------------------------------------------------------------------------------*/
static void* checkedMalloc(size_t nbytes, const char* name)
{
	void* p = malloc(nbytes);
	if (p == NULL) {
		printf("   Memory allocation failed for %s.\n", name);
	}
	return p;
}

/* checkedAlignedMalloc --------------------------------------------------------
 * Allocate 32-byte aligned host memory for SIMD buffers.
 * Args   : size_t nbytes             I   number of bytes to allocate
 *          const char* name          I   buffer name for error reporting
 * Return : void*                        allocated aligned pointer
 *------------------------------------------------------------------------------*/
static void* checkedAlignedMalloc(size_t nbytes, const char* name)
{
	void* p = _mm_malloc(nbytes, 32);
	if (p == NULL) {
		printf("   Aligned memory allocation failed for %s.\n", name);
	}
	return p;
}

/* gatherCode16 ----------------------------------------------------------------
 * Gather 16 local-code chips from the int32 code table and repack them
 * into one AVX register of 16 x int16 samples for the existing correlator path.
 * Args   : const int* caCode         I   int32 code table for one PRN
 *          __m256i Index_reg1        I   code indexes for samples 0..7
 *          __m256i Index_reg2        I   code indexes for samples 8..15
 * Return : __m256i                      gathered code chips in chronological order
 *------------------------------------------------------------------------------*/
static __m256i gatherCode16(const int* caCode, __m256i Index_reg1, __m256i Index_reg2)
{
	__m256i EPL_reg1 = _mm256_i32gather_epi32(caCode, Index_reg1, 4);
	__m256i EPL_reg2 = _mm256_i32gather_epi32(caCode, Index_reg2, 4);
	__m256i EPL_reg = _mm256_packs_epi32(EPL_reg1, EPL_reg2);
	return _mm256_permute4x64_epi64(EPL_reg, 0xD8);
}

/* checkInputs ----------------------------------------------------------------
 * Validate the fixed interface values once, before persistent buffers are
 * allocated. Later calls reuse the initialized configuration.
 * Return : 1 for valid input; 0 otherwise
 *------------------------------------------------------------------------------*/
static int checkInputs(int fileType, int blksize, const short* rawSignal,
	const int* caCode, int codeLen)
{
	if (rawSignal == NULL || caCode == NULL || blksize <= 0 || codeLen <= 0 ||
		(fileType != 1 && fileType != 2)) {
		printf("   Invalid input arguments for corrSIMDSerialBPSK.\n");
		return 0;
	}
	return 1;
}

/* cleanup --------------------------------------------------------------------
 * Release all persistent host buffers. It is safe to call repeatedly.
 * Return : None
 *------------------------------------------------------------------------------*/
void cleanup(void)
{
	printf("   SIMD DLL is terminating, destroying allocated memory ...\n");

	/* --------------- Free memory --------------- */
	if (localCode != NULL) {
		for (int i = 0; i < 3; ++i) {
			if (localCode[i] != NULL) {
				_mm_free(localCode[i]);
				localCode[i] = NULL;
			}
		}
		free(localCode);
		localCode = NULL;
	}

	if (iBasebandSignal != NULL) { _mm_free(iBasebandSignal); iBasebandSignal = NULL; }
	if (qBasebandSignal != NULL) { _mm_free(qBasebandSignal); qBasebandSignal = NULL; }
	if (rawSignalI != NULL) { _mm_free(rawSignalI); rawSignalI = NULL; }
	if (rawSignalQ != NULL) { _mm_free(rawSignalQ); rawSignalQ = NULL; }
	initialized = 0;
}
