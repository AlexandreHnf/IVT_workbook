#include <iostream>
#include <math.h>
#include <fstream>
#include <random>
#include <sstream>
#include "functions.cpp"

using namespace std;

typedef float img[];
typedef float img256[256*256];
typedef float img8[8*8];
typedef float img32[32*32];

typedef vector<float> vecf;
typedef vector<vector<float>> vecf2;

#define TEST_LOAD_STORE "test_files/test_load_store.raw"
#define TEST_LOAD_STORE_TXT "test_files/test_load_store_txt.txt"
#define TEST_LOAD_STORE_BITSTR "test_files/test_load_store_bitstream.txt"

// SESSION 1
#define FILE_LENA "lena_256x256.raw"
#define FILE_COS_PATTERN  "session1/img_cos_pattern.raw"
#define FILE_COS_PATTERN_LENA  "session1/modified_lena_256x256.raw"

// SESSION 2
#define FILE_UNIFORM  "session2/uniform_image.raw"
#define FILE_GAUSSIAN  "session2/gaussian_image.raw"
#define FILE_UNIFORM_LENA  "session2/uniform_lena_256x256.raw"
#define FILE_GAUSSIAN_LENA  "session2/gaussian_lena_256x256.raw"
#define FILE_BLURRY_LENA  "session2/blurry_lena_generated_manually.raw"
#define FILE_GAUSSIAN_BLURRY_LENA_5_2  "session2/blurry_gaussian_high_lena_256x256_5_2.raw"
#define FILE_GAUSSIAN_LENA_5_3  "session2/gaussian_lena_256x256_5_3.raw"
#define FILE_GAUSSIAN_LENA_5_3_1  "session2/gaussian_lena_256x256_5_3_1.raw"
#define FILE_GAUSSIAN_LENA_5_3_1_5  "session2/gaussian_lena_256x256_5_3_1_5.raw"

// SESSION 3
#define FILE_DCT_MATRIX  "session3/dct_matrix.raw"
#define FILE_IDCT_MATRIX  "session3/idct_matrix.raw"
#define FILE_DCT_LENA  "session3/dct_lena.raw"
#define FILE_THRESHOLD_DCT_LENA  "session3/threshold_dct_lena10.raw"
#define FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_10  "session3/rec_threshold_dct_lena10.raw"
#define FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_5  "session3/rec_threshold_dct_lena5.raw"
#define FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_20  "session3/rec_threshold_dct_lena20.raw"
#define FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_2  "session3/rec_threshold_dct_lena2.raw"
#define FILE_RECONSTRUCTED_DCT_LENA  "session3/rec_dct_lena.raw"
#define FILE_RECONSTRUCTED_DCT_LENA2  "session3/rec_dct_lena2.raw"

// SESSION 4
#define FILE_QUANTIZATION  "session4/quantization.raw"
#define FILE_8BPP_LENA  "session4/8bpp_lena.raw"
#define FILE_DCT_MATRIX8  "session4/dct_matrix8.raw"
#define FILE_DCT_8_2  "session4/8_2_1_lena_dct.raw"
#define FILE_QDCT_8_2  "session4/8_2_2_lena_Qdct.raw"
#define FILE_IQDCT_8_2  "session4/8_2_3_lena_IQdct.raw"
#define FILE_IQIDCT_8_2  "session4/8_2_4_lena_IQidct.raw"
#define FILE_IQIDCT_8_2_8bpp "session4/8_4_lena_8bpp.raw"
#define FILE_ENCODED_LENA8_5  "session4/8_5_encoded_lena.raw"
#define FILE_ENCODED_LENA8_5_t  "session4/8_5_t_encoded_lena.raw"
#define FILE_DECODED_LENA8_5  "session4/8_5_decoded_lena.raw"
#define FILE_CONTIGUOUS_9 "session4/9_contiguous_lena.raw"
#define FILE_INTERLEAVED_9 "session4/9_interleaved_lena.raw"
#define FILE_INV_INTERLEAVED_9 "session4/9_inv_interleaved_lena.raw"

// SESSION 5
#define FILE_32_QDCT "session5/qdct_lena32x32_10_1.raw"
#define FILE_DELTA_DC_TXT "session5/delta_dct_10_2.txt"
#define FILE_RECONSTRUCTED_DELTA "session5/reconstructed_delta_dct_10_3.raw"
#define FILE_DC_32_11_1 "session5/DC_image_11_1.raw"
#define FILE_11_2_TXT "session5/11_2_txt.txt"
#define FILE_11_2_TXT_2 "session5/11_2_noRLE_txt.txt"
#define FILE_DECODED_11_3 "session5/decoded_lena_11_3.raw"
#define FILE_TEST_RECONSTRUCTED_INTERLEAVED "session5/test_reconstructed_interleaved.raw"
#define FILE_12_2_NORMALIZED "session5/12_2_normalized.txt"
#define FILE_12_2_OCC "session5/12_2_occ.txt"

// SESSION 6
#define FILE_BITSTREAM_AC_14_1 "session6/bitstream_AC_14_1.txt"
#define FILE_BITSTREAM_DC_14_1 "session6/bitstream_DC_14_1.txt"
#define FILE_DELTA_DC_14_1 "session6/delta_encoded_DC_14_1.txt"
#define FILE_DECOMPRESSED_LENA_14_2 "session6/decompressed_lena_13_2.raw"
#define FILE_CLIP_14_2 "session6/clipped_decompressed_14_2.raw"



void ex2_2() {
    cout << "----- 2.2" << endl;
    img256 img_cos_pattern;
    generateCosPatternImage(img_cos_pattern, 256);
    //displayImage(img_cos_pattern, "cos pattern img created : ");
    store(FILE_COS_PATTERN, img_cos_pattern, 256*256);
}

void ex3_3() {
    cout << "----- 3.3" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    //displayImage(lena, "Lena raw loaded : ");

    img256 img_cos_pattern;
    generateCosPatternImage(img_cos_pattern, 256);
    img256 modified_lena;
    load(FILE_LENA, modified_lena, 256*256);
    imageProduct(modified_lena, img_cos_pattern, 256);
    //displayImage(modified_lena, "Lena raw with cosine pattern : ");
    // Store modified lena
    store(FILE_COS_PATTERN_LENA, modified_lena, 256*256);
}

void ex3_4() {
    cout << "----- 3.4" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img256 modified_lena;
    load(FILE_COS_PATTERN_LENA, modified_lena, 256*256);
    float MSE = mse(lena, modified_lena, 256);
    cout << "MSE : " << MSE << endl;
}

void ex3_5() {
    cout << "----- 3.5" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img256 modified_lena;
    load(FILE_COS_PATTERN_LENA, modified_lena, 256*256);
    float PSNR = psnr(lena, modified_lena, 255, 256);
    cout << "PSNR :" << PSNR << endl;
}

void session1() {
    // =========================================================================================
    // ====================================== SESSION 1 ========================================
    // ========================================================================================

    cout << "SESSION 1" << endl;

    // ============================= [SESSION 1 ex 2.1 & 2.2]
    ex2_2();

    // =============================  [SESSION 1 ex 3.1 & 3.2 & 3.3]
    ex3_3();

    // =============================  [SESSION 1 ex 3.4]
    ex3_4();

    // =============================  [SESSION 1 ex 3.5]
    ex3_5();

}

void ex4_1() {
    cout << "----- 4.1" << endl;
    img256 uniform_image;
    // generate image with uniform distribution
    generateUDRN(uniform_image, -0.5, 0.5, 256);
    store(FILE_UNIFORM, uniform_image, 256*256);
    // what is the expected MSE of the random image, compared to the expected mean 0
    img256 zero_image;
    generateZeroImage(zero_image, 256);
    float uniform_mse = mse(uniform_image, zero_image, 256);
    cout << "uniform MSE : " << uniform_mse << endl;
}

void ex4_2() {
    cout << "----- 4.2" << endl;
    img256 gaussian_image;
    // generate image with gaussian distribution
    generateGDRN(gaussian_image, 0.0, sqrt(0.0830861), 256);
    store(FILE_GAUSSIAN, gaussian_image, 256*256);
    // what is the expected MSE of the random image, compared to the expected mean 0
    img256 zero_image;
    generateZeroImage(zero_image, 256);
    float gaussian_mse = mse(gaussian_image, zero_image, 256);
    cout << "gaussian MSE : " << gaussian_mse << endl;
}

void ex4_4() {
    cout << "----- 4.4" << endl;
    img256 uniform_image;
    load(FILE_UNIFORM, uniform_image, 256*256);
    img256 uniform_lena;
    load(FILE_LENA, uniform_lena, 256*256);
    imageAddition(uniform_lena, uniform_image, 256); // change to "add function"
    store(FILE_UNIFORM_LENA, uniform_lena, 256*256);

    img256 gaussian_image;
    load(FILE_GAUSSIAN, gaussian_image, 256*256);
    img256 gaussian_lena;
    load(FILE_LENA, gaussian_lena, 256*256);
    imageAddition(gaussian_lena, gaussian_image, 256); // change to "add function"
    store(FILE_GAUSSIAN_LENA, gaussian_lena, 256*256);
}

void ex5_1() {
    cout << "----- 5.1" << endl;
    img256 original_lena;
    load(FILE_LENA, original_lena, 256*256);
    img256 blurry_lena;
    load(FILE_BLURRY_LENA, blurry_lena, 256*256);
    float blurry_lena_mse = mse(blurry_lena, original_lena, 256);
    cout << "MSE blurry lena: " << blurry_lena_mse << endl;
}

void ex5_2() {
    cout << "----- 5.2" << endl;
    img256 high_gaussian_image;
    generateGDRN(high_gaussian_image, 0.0, 10, 256); // gaussian image with high variance
    img256 gaussian_lena;
    load(FILE_BLURRY_LENA, gaussian_lena, 256*256);
    imageAddition(gaussian_lena, high_gaussian_image, 256); // blurry lena + high variance gaussian
    store(FILE_GAUSSIAN_BLURRY_LENA_5_2, gaussian_lena, 256*256);
}

void ex5_3(float sigma, string filename) {
    cout << "----- 5.3" << endl;
    img256 gaussian_image;
    generateGDRN(gaussian_image, 0.0, sigma, 256); // gaussian image with variance = mse blurry lena
    img256 gaussian_lena;
    load(FILE_LENA, gaussian_lena, 256*256);
    imageAddition(gaussian_lena, gaussian_image, 256); // Lena + gaussian
    store(filename, gaussian_lena, 256*256);

    img256 original_lena;
    load(FILE_LENA, original_lena, 256*256);
    float gaussian_lena_mse = mse(gaussian_lena, original_lena, 256);
    cout << "MSE gaussian lena with same mse than blurry lena : " << gaussian_lena_mse << endl;
}

void ex5_4(string other_gaussian_file, float sigma) {
    cout << "----- 5.4" << endl;

    img256 gaussian_lena3;
    load(FILE_GAUSSIAN_LENA_5_3, gaussian_lena3, 256*256);
    img256 blurry_gaussian_lena3;
    load(other_gaussian_file, blurry_gaussian_lena3, 256*256);
    float blurry_gaussian3_mse = mse(blurry_gaussian_lena3, gaussian_lena3, 256);
    cout << "MSE 5.4 sigma = " << sigma << ": " << blurry_gaussian3_mse << endl;
}

void session2() {
    // =========================================================================================
    // ====================================== SESSION 2 ========================================
    // =========================================================================================

    cout << "SESSION 2" << endl;

    // ===========  [SESSION 2 ex 4.1]
    ex4_1();

    // =========generate image with gaussian distribution [SESSION 2 ex 4.2]
    ex4_2();

    // =========lena with uniform and gaussian [SESSION 2 ex 4.4]
    ex4_4();

    // =========lena blurry [SESSION 2 ex 5.1]
    ex5_1();

    // =========add gaussian distribution to blurry lena [SESSION 2 ex 5.2]
    ex5_2();

    // =========add gaussian to original lena [SESSION 2 ex 5.3]
    ex5_3(sqrt(40.7731), FILE_GAUSSIAN_LENA_5_3);
    ex5_3(1, FILE_GAUSSIAN_LENA_5_3_1);
    ex5_3(1.5, FILE_GAUSSIAN_LENA_5_3_1_5);

    // [SESSION 2 ex 5.4]
    ex5_4(FILE_GAUSSIAN_LENA_5_3_1, 1);
    ex5_4(FILE_GAUSSIAN_LENA_5_3_1_5, 1.5);

}

void ex6_1() {
    cout << "----- 6.1" << endl;
    img256 dct_mat;
    createDCTmatrix(dct_mat, 256);
    store(FILE_DCT_MATRIX, dct_mat, 256*256);

}

void ex6_2() {
    cout << "----- 6.2" << endl;
    img256 dct_mat;
    createDCTmatrix(dct_mat, 256);
    float DC_coefficient = dct_mat[0];
    for (int i=0; i<256; i++){ // all elements of the basic vector of the DC coefficient
        if (dct_mat[i] != DC_coefficient) {
            cout << "all elements of the basis vector for the DC coefficient are NOT all identical" << endl;
            return;
        }
    }
    cout << "all elements of the basis vector for the DC coefficient are all identical" << endl;
}

void ex6_3() {
    cout << "----- 6.3" << endl;
    // Check that the matrix is orthonormal :
    img256 dct;
    createDCTmatrix(dct, 256);
    img256 idct; // IDCT
    createIDCTmatrix(idct, 256);
    bool iso = isOrthonormal(idct, dct, 256);
    cout << "is orthogonal ? " << iso << endl;
}

void ex7_1() {
    cout << "----- 7.1" << endl;
    img256 original_lena;
    load(FILE_LENA, original_lena, 256*256);
    img256 res;
    transform(original_lena, res, 256);
    store(FILE_DCT_LENA, res, 256*256);
}

void ex7_2() {
    cout << "----- 7.2" << endl;
    img256 dct_lena;
    load(FILE_DCT_LENA, dct_lena, 256*256);
    threshold(dct_lena, 10, 256);
    store(FILE_THRESHOLD_DCT_LENA, dct_lena, 256*256);
}

void ex7_3_1() {
    cout << "----- 7.3.1" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img256 transformed;
    transform(lena, transformed, 256);
    img256 reconstructed;
    inverseTransform(transformed, reconstructed, 256);
    store(FILE_RECONSTRUCTED_DCT_LENA, reconstructed, 256*256);
}

void ex7_3_thresh(string filename, float t) {
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img256 transformed;
    transform(lena, transformed, 256);
    threshold(transformed, t, 256);
    img256 reconstructed;
    inverseTransform(transformed, reconstructed, 256);
    float psnr_val = psnr(reconstructed, lena, 255, 256);
    cout << "PSNR reconstructed lena with threshold " << t << " : " << psnr_val << endl;
    store(filename, reconstructed, 256*256);
}

void ex7_3_2() {
    cout << "----- 7.3.2" << endl;
    ex7_3_thresh(FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_2, 2);
    ex7_3_thresh(FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_5, 5);
    ex7_3_thresh(FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_10, 10);
    ex7_3_thresh(FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_20, 20);
}

void session3() {

    // =========================================================================================
    // ====================================== SESSION 3 ========================================
    // =========================================================================================

    cout << "SESSION 3" << endl;
    // [SESSION 3 ex 6.1]
    ex6_1();

    // [SESSION 3 ex 6.2]
    ex6_2();

    // [SESSION 3 ex 6.3]
    ex6_3();

    // [SESSION 3 ex 7.1]
    ex7_1();

    // [SESSION 3 ex 7.2]
    ex7_2();

    // [SESSION 3 ex 7.3]
    ex7_3_1(); // test transform without threshold

    ex7_3_2(); // transform with threshold
}



void ex8_1() {
    cout << "----- 8.1" << endl;
    img8 Qtable;
    getQtable(Qtable, 8);
    store(FILE_QUANTIZATION, Qtable, 8*8);
}

void ex8_2() {
    cout << "----- 8.2" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img256 lena_dct;
    img256 lena_Qdct;
    img256 lena_IQdct;
    img256 lena_IQidct;
    img8 Qtable;
    getQtable(Qtable, 8);
    approximate(lena, lena_dct, lena_Qdct, lena_IQdct, lena_IQidct, Qtable, 256, 8);
    store(FILE_DCT_8_2, lena_dct, 256*256);
    store(FILE_QDCT_8_2, lena_Qdct, 256*256);
    store(FILE_IQDCT_8_2, lena_IQdct, 256*256);
    store(FILE_IQIDCT_8_2, lena_IQidct, 256*256);
}

void ex8_3() {
    cout << "----- 8.3" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    clip(FILE_8BPP_LENA, lena, 256);
}

void ex8_4() {
    cout << "----- 8.4" << endl;
    img256 approximated_lena;
    load(FILE_IQIDCT_8_2, approximated_lena, 256*256);
    clip(FILE_IQIDCT_8_2_8bpp, approximated_lena, 256);
}

void ex8_5() {
    cout << "----- 8.5" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img256 encoded_lena;
    img256 decoded_lena;
    img8 Qtable;
    getQtable(Qtable, 8);

    encode(lena, encoded_lena, Qtable, 256, 8);
    store(FILE_ENCODED_LENA8_5, encoded_lena, 256*256);
    decode(encoded_lena, decoded_lena, Qtable, 256, 8);
    store(FILE_DECODED_LENA8_5, decoded_lena, 256*256);
}

void ex9_1_1() {
    cout << "----- 9.1.1" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img256 contiguous_dct_lena;
    img8 Qtable;
    getQtable(Qtable, 8);

    encode(lena, contiguous_dct_lena, Qtable, 256, 8);
    store(FILE_INTERLEAVED_9, contiguous_dct_lena, 256*256);
}

void ex9_1_2() {
    cout << "----- 9.1.2" << endl;
    // transformm a 256x256 pixels layout with 32x32 grid of 8x8 contiguous DCT coefs :
    img256 contiguous_dct_lena;
    load(FILE_DCT_8_2, contiguous_dct_lena, 256*256);
    img256 interleaved_dct_lena;
    interleavedLayout(contiguous_dct_lena, interleaved_dct_lena, 8, 256);
    store(FILE_INTERLEAVED_9, interleaved_dct_lena, 256*256);

    // inverse interleave :
    img256 inv_interleaved;
    interleavedLayout(interleaved_dct_lena, inv_interleaved, 32, 256);
    store(FILE_INV_INTERLEAVED_9, inv_interleaved, 256*256);
}


void session4() {

    // =========================================================================================
    // ====================================== SESSION 3 ========================================
    // =========================================================================================

    cout << "SESSION 4" << endl;

    // [SESSION 4 ex 8.1]
    ex8_1();

    // [SESSION 4 ex 8.2]
    ex8_2();

    // [SESSION 4 ex 8.3]
    ex8_3();

    // [SESSION 4 ex 8.4]
    ex8_4();

    // [SESSION 4 ex 8.5]
    ex8_5();

    // [SESSION 4 ex 9.1]
    ex9_1_1();
    ex9_1_2();

}

void ex10_1() {
    cout << "----- 10.1" << endl;
    // create a 32x32 pixels image from quantized DC terms of each 8x8 pixels block
    img256 quantized_DC;
    load(FILE_QDCT_8_2, quantized_DC, 256*256);
    img32 res;
    quantizedDCtermsMat(quantized_DC, res, 8, 256);
    store(FILE_32_QDCT, res, 32*32);
}

void ex10_2() {
    cout << "----- 10.2" << endl;
    img32 QDCTmat;
    load(FILE_32_QDCT, QDCTmat, 32*32);
    deltaEncoding(QDCTmat, 32);
    storeTXTmatrix(FILE_DELTA_DC_TXT, QDCTmat, 32*32);
}

void ex10_3() {
    cout << "----- 10.3" << endl;
    img32 delta_QDCterms;
    loadTXTmatrix(FILE_DELTA_DC_TXT, delta_QDCterms, 32*32);
    deltaDecoding(delta_QDCterms, 32);
    store(FILE_RECONSTRUCTED_DELTA, delta_QDCterms, 32*32);
}

void ex11_1() {
    cout << "----- 11.1" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img256 contiguous_qdct_lena;
    img8 Qtable;
    getQtable(Qtable, 8);
    encode(lena, contiguous_qdct_lena, Qtable, 256, 8);

    img256 interleaved_qdct_lena;
    interleavedLayout(contiguous_qdct_lena, interleaved_qdct_lena, 8, 256);
    veci zigzag_pattern = zigzagPattern(256);
    veci DC_indices = getDCIndicesFromInterleaved(32, 256);
    cout << "nb of DC : " << DC_indices.size() << endl;
    veci AC_pattern = getACzigzagPattern(getDCIndicesFromInterleaved(32, 256), zigzagPattern(256), 256);
    veci DC_pattern = getDCzigzagPattern(getDCIndicesFromInterleaved(32, 256), zigzagPattern(256), 256);
    cout << "nb of AC : " << AC_pattern.size() << endl;
}

vecf2 ex11_2() {
    cout << "----- 11.2" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    // showRunLengths(encodedRLE);
    storeRLE(FILE_11_2_TXT, encodedRLE);
    return encodedRLE;
}

void ex11_3() {
    cout << "----- 11.3" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    img32 deltaEncodedDC;
    loadTXTmatrix(FILE_DELTA_DC_TXT, deltaEncodedDC, 32*32);

    img256 decoded;
    decodeRLE(encodedRLE, decoded, deltaEncodedDC, 256, 8);
    store(FILE_DECODED_11_3, decoded, 256*256);
}

void ex12_1() {
    cout << "----- 12.1" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    vecf P = occurences(encodedRLE);
    int N = P.size();
    float to_store[N];
    for (int i=0; i<N; i++) {to_store[i] = P[i];}
    storeTXTmatrix(FILE_12_2_OCC, to_store, N);
    cout << "longuest run length N produced : " << P.size()-1 << endl;
    cout << "nb of runs M necessary :" << encodedRLE.size() << endl;
}

void ex12_2() {
    cout << "----- 12.2" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    vecf P = occurences(encodedRLE);
    vecf normalized_encoded = normalizeOccurences(P);
    int N = normalized_encoded.size();
    float norm[N];
    for (int i=0; i<N; i++) {norm[i] = normalized_encoded[i];}
    storeTXTmatrix(FILE_12_2_NORMALIZED, norm, N);
}

void ex12_3() {
    cout << "----- 12.3" << endl;
    img256 lena;
    load(FILE_LENA, lena, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    vecf P = occurences(encodedRLE);
    vecf normalized_encoded = normalizeOccurences(P);
    float H = entropy(normalized_encoded);
    cout << "entropy : " << H << endl;
    int nb_AC = 256*256 - (256/8)*(256/8);
    cout << "minimum possible file size for encoding AC coefficients with RLE : " << H*nb_AC << endl;
}


void session5() {

    // =========================================================================================
    // ====================================== SESSION 5 ========================================
    // =========================================================================================

    cout << "SESSION 5" << endl;

    // [SESSION 5 ex 10.1]
    ex10_1();

    // [SESSION 5 ex 10.2]
    ex10_2();

    // [SESSION 5 ex 10.3]
    ex10_3();

    // [SESSION 5 ex 11.1]
    ex11_1();

    // [SESSION 5 ex 11.2]
    ex11_2();

    // [SESSION5 ex 11.3]
    ex11_3();

    // [SESSION 5 ex 12.1]
    ex12_1();

    // [SESSION 5 ex 12.2]
    ex12_2();

    // [SESSION 5 ex 12.3]
    ex12_3();
}

void ex14_1() {
    cout << "----- 14.1" << endl;

    img256 lena;
    load(FILE_LENA, lena, 256*256);
    compress(FILE_BITSTREAM_DC_14_1, FILE_BITSTREAM_AC_14_1, lena, 256, 8);
}

void ex14_2() {
    cout << "----- 14.2" << endl;
    string compressed_bitstream = loadBitstream(FILE_BITSTREAM_AC_14_1);
    img32 deltaEncodedDC;
    loadTXTmatrix(FILE_DELTA_DC_14_1, deltaEncodedDC, 32*32);
    string bitstream_DC = loadBitstream(FILE_BITSTREAM_DC_14_1);
    img32 deltaEncodedDC2;
    bitstreamDCToDelta(bitstream_DC, deltaEncodedDC2);
//    assert((std::equal(std::begin(deltaEncodedDC), std::end(deltaEncodedDC), std::begin(deltaEncodedDC2)) == true));

    img256 decompressed_lena;
    decompress(FILE_BITSTREAM_DC_14_1, FILE_BITSTREAM_AC_14_1, decompressed_lena, 256, 8);
    store(FILE_DECOMPRESSED_LENA_14_2, decompressed_lena, 256*256);
    clip(FILE_CLIP_14_2, decompressed_lena, 256);
}

void session6() {

    // =========================================================================================
    // ====================================== SESSION 5 ========================================
    // =========================================================================================

    cout << "SESSION 6" << endl;

    // [SESSION 6 ex 14.1]
    ex14_1();

    // [SESSION 6 ex 14.2]
    ex14_2();
}

void testing() {
    cout << "----- starting tests " << endl;
    int i = 0;
    TEST_store_load(TEST_LOAD_STORE); i++;
    TEST_imageProduct(); i++;
    TEST_imageAddition(); i++;
    TEST_mse(); i++;
    TEST_psnr(); i++;
    TEST_transposeSquareMatrix(); i++;
    TEST_squareMatrixMultiplication(); i++;
    TEST_isIdentityMatrix(); i++;
    TEST_isOrthonormal(); i++;
    TEST_interleavedLayout(); i++;
    TEST_deltaEncoding(); i++;
    TEST_deltaDecoding(); i++;
    TEST_store_load_txt(TEST_LOAD_STORE_TXT); i++;
    TEST_zigzagPattern(); i++;
    TEST_getDCIndicesFromInterleaved(); i++;
    TEST_getDCzigzagPattern(); i++;
    TEST_getACzigzagPattern(); i++;
    TEST_getBackACDCzigzag(); i++;
    TEST_runLengthEncoding(); i++;
    TEST_occurences(); i++;
    TEST_runLengthDecoding(); i++;
    TEST_normalizeOccurences(); i++;
    TEST_entropy(); i++;
    TEST_decToBinary(); i++;
    TEST_binaryToDec(); i++;
    TEST_golomb(); i++;
    TEST_inverseGolombStream(); i++;
    TEST_mapValueForGolomb(); i++;
    TEST_reverseMapForGolomb(); i++;
    TEST_store_load_bitstream(TEST_LOAD_STORE_BITSTR); i++;
    TEST_generateBitStreamAC(); i++;
    TEST_bitstreamACToRLE(); i++;
    cout << "All tests (" << i << ") passed successfully !" << endl;
}

int main() {
    std::cout << " ================ Staaart == " << std::endl;

    session1();

    session2();

    session3();

    session4();

    session5();

    session6();

    testing();

    std::cout << " ================ Finish == " << std::endl;
    return 0;
}
