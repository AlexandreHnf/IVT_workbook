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

// TODO : rename each file with IVT2020_Heneffe_Alexandre
#define TEST_LOAD_STORE "test_files/test_load_store.raw"
#define TEST_LOAD_STORE_TXT "test_files/test_load_store_txt.txt"
#define TEST_LOAD_STORE_BITSTR "test_files/test_load_store_bitstream.txt"

#define FILE_LENA "lena_256x256.raw"

// SESSION 1
#define FILE_COS_PATTERN  "session1/2_2_img_cos_pattern.raw"
#define FILE_COS_PATTERN_LENA  "session1/3_3_modified_lena_256x256.raw"

// SESSION 2
#define FILE_UNIFORM  "session2/4_1_uniform_image.raw"
#define FILE_GAUSSIAN  "session2/4_2_gaussian_image.raw"
#define FILE_UNIFORM_LENA  "session2/4_4_1_uniform_lena_256x256.raw"
#define FILE_GAUSSIAN_LENA  "session2/4_4_2_gaussian_lena_256x256.raw"
#define FILE_BLURRY_LENA  "session2/5_1_blurry_lena_generated_manually.raw"
#define FILE_GAUSSIAN_BLURRY_LENA_5_2  "session2/5_2_blurry_gaussian_high_lena_256x256.raw"
#define FILE_GAUSSIAN_LENA_5_3  "session2/5_3_gaussian_lena_256x256.raw"
#define FILE_GAUSSIAN_LENA_5_4_1  "session2/5_4_gaussian_lena_1_generated_manually.raw"
#define FILE_GAUSSIAN_LENA_5_4_1_5  "session2/5_4_gaussian_lena_1_5_generated_manually.raw"

// SESSION 3
#define FILE_DCT_MATRIX  "session3/6_1_dct_matrix.raw"
#define FILE_IDCT_MATRIX  "session3/6_3_idct_matrix.raw"
#define FILE_DCT_LENA  "session3/7_1_dct_lena.raw"
#define FILE_THRESHOLD_DCT_LENA  "session3/7_2_threshold_dct_lena10.raw"
#define FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_10  "session3/7_3_rec_threshold_dct_lena10.raw"
#define FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_5  "session3/7_3_rec_threshold_dct_lena5.raw"
#define FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_20  "session3/7_3_rec_threshold_dct_lena20.raw"
#define FILE_RECONSTRUCTED_THRESHOLD_DCT_LENA_2  "session3/7_3_rec_threshold_dct_lena2.raw"
#define FILE_RECONSTRUCTED_DCT_LENA  "session3/7_3_rec_dct_lena.raw"

// SESSION 4
#define FILE_QUANTIZATION  "session4/8_1_quantization.raw"
#define FILE_8BPP_LENA  "session4/8_3_8bpp_lena.raw"
#define FILE_DCT_8_2  "session4/8_2_1_lena_dct.raw"
#define FILE_QDCT_8_2  "session4/8_2_2_lena_Qdct.raw"
#define FILE_IQDCT_8_2  "session4/8_2_3_lena_IQdct.raw"
#define FILE_IQIDCT_8_2  "session4/8_2_4_lena_IQidct.raw"
#define FILE_QDCT_8_2_8bpp "session4/8_4_approx_lena_8bpp.raw"
#define FILE_IQIDCT_8_2_8bpp "session4/8_4_dec_lena_8bpp.raw"
#define FILE_ENCODED_LENA8_5  "session4/8_5_encoded_lena.raw"
#define FILE_DECODED_LENA8_5  "session4/8_5_decoded_lena.raw"
#define FILE_CONTIGUOUS_9 "session4/9_contiguous_lena.raw"
#define FILE_INTERLEAVED_9 "session4/9_interleaved_lena.raw"
#define FILE_INV_INTERLEAVED_9 "session4/9_inv_interleaved_lena.raw"

// SESSION 5
#define FILE_32_QDCT "session5/10_1_qdct_lena32x32.raw"
#define FILE_AVERAGE_DOWNSIZED "session5/10_1_downsized_average_lena32_generated_manually.raw"
#define FILE_DELTA_DC_TXT "session5/10_2_delta_dct.txt"
#define FILE_RECONSTRUCTED_DELTA "session5/10_3_reconstructed_delta_dct.raw"
#define FILE_11_2_TXT "session5/11_2_RLE.txt"
#define FILE_DECODED_11_3 "session5/11_3_decoded_lena.raw"
#define FILE_12_2_NORMALIZED "session5/12_2_normalized.txt"
#define FILE_12_2_OCC "session5/12_2_occ.txt"

// SESSION 6
#define FILE_BITSTREAM_AC_14_1 "session6/14_1_bitstream_AC.txt"
#define FILE_BITSTREAM_DC_14_1 "session6/14_1_bitstream_DC.txt"
#define FILE_DELTA_DC_14_1 "session6/14_1_delta_encoded_DC.txt"
#define FILE_DECOMPRESSED_LENA_14_2 "session6/14_2_decompressed_lena.raw"
#define FILE_CLIP_14_2 "session6/14_2_clipped_decompressed.raw"



void ex2_2() {
    cout << "----- 2.2" << endl;
    vecf img_cos_pattern = generateCosPatternImage(256);
    //displayImage(img_cos_pattern, "cos pattern img created : ");
    store(FILE_COS_PATTERN, img_cos_pattern);
}

void ex3_3() {
    cout << "----- 3.3" << endl;

    vecf img_cos_pattern = generateCosPatternImage(256);
    vecf lena = load(FILE_LENA, 256*256);
    vecf cos_lena = imageProduct(lena, img_cos_pattern, 256);
    // Store modified lena
    store(FILE_COS_PATTERN_LENA, cos_lena);
}

void ex3_4() {
    cout << "----- 3.4" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    vecf cos_lena = load(FILE_COS_PATTERN_LENA, 256*256);
    float MSE = mse(lena, cos_lena, 256);
    cout << "MSE : " << MSE << endl;
}

void ex3_5() {
    cout << "----- 3.5" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    vecf modified_lena = load(FILE_COS_PATTERN_LENA, 256*256);
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
    // generate image with uniform distribution
    vecf uniform_image = generateUDRN(-0.5, 0.5, 256);
    store(FILE_UNIFORM, uniform_image);
    // what is the expected MSE of the random image, compared to the expected mean 0
    vecf zero_img(256*256, 0);
    float uniform_mse = mse(uniform_image, zero_img, 256);
    cout << "uniform MSE : " << uniform_mse << endl;
}

void ex4_2() {
    cout << "----- 4.2" << endl;
    // generate image with gaussian distribution
    vecf gaussian_image = generateGDRN(0.0, sqrt(0.0830861), 256);
    store(FILE_GAUSSIAN, gaussian_image);
    // what is the expected MSE of the random image, compared to the expected mean 0
    vecf zero_img(256*256, 0);
    float gaussian_mse = mse(gaussian_image, zero_img, 256);
    cout << "gaussian MSE : " << gaussian_mse << endl;
}

void ex4_4() {
    cout << "----- 4.4" << endl;
    vecf original_lena = load(FILE_LENA, 256*256);
    vecf uniform_image = load(FILE_UNIFORM, 256*256);
    vecf lena_u = load(FILE_LENA, 256*256);
    vecf uniform_lena = imageAddition(lena_u, uniform_image, 256);
    store(FILE_UNIFORM_LENA, uniform_lena);
    cout << "MSE lena - uniform lena = " << mse(original_lena, uniform_lena, 256) << endl;
    cout << "PSNR lena - uniform lena = " << psnr(original_lena, uniform_lena, 255, 256) << endl;

    vecf gaussian_image = load(FILE_GAUSSIAN, 256*256);
    vecf lena_g = load(FILE_LENA, 256*256);
    vecf gaussian_lena = imageAddition(lena_g, gaussian_image, 256);
    store(FILE_GAUSSIAN_LENA, gaussian_lena);
    cout << "MSE lena - gaussian lena = " << mse(original_lena, gaussian_lena, 256) << endl;
    cout << "PSNR lena - gaussian lena = " << psnr(original_lena, gaussian_lena, 255, 256) << endl;
}

void ex5_1() {
    cout << "----- 5.1" << endl;
    vecf original_lena = load(FILE_LENA, 256*256);
    vecf blurry_lena = load(FILE_BLURRY_LENA, 256*256);
    float blurry_lena_mse = mse(blurry_lena, original_lena, 256);
    cout << "MSE blurry lena: " << blurry_lena_mse << endl;
    cout << "PSNR blurry lena : " << psnr(blurry_lena, original_lena, 255, 256) << endl;
}

void ex5_2() {
    cout << "----- 5.2" << endl;
    vecf high_gaussian_image = generateGDRN(0.0, 10, 256); // gaussian image with high variance
    vecf blurry_lena = load(FILE_BLURRY_LENA, 256*256);
    vecf gaussian_blurry_lena = imageAddition(blurry_lena, high_gaussian_image, 256); // blurry lena + high variance gaussian
    store(FILE_GAUSSIAN_BLURRY_LENA_5_2, gaussian_blurry_lena);
}

void ex5_3() {
    cout << "----- 5.3" << endl;
    vecf gaussian_image = generateGDRN(0.0, sqrt(40.7731), 256); // gaussian image with variance = mse blurry lena
    vecf lena = load(FILE_LENA, 256*256);
    vecf gaussian_lena = imageAddition(lena, gaussian_image, 256); // Lena + gaussian
    store(FILE_GAUSSIAN_LENA_5_3, gaussian_lena);

    vecf original_lena = load(FILE_LENA, 256*256);
    float gaussian_lena_mse = mse(gaussian_lena, original_lena, 256);
    cout << "MSE gaussian lena with same mse than blurry lena : " << gaussian_lena_mse << endl;
}

void ex5_4(string other_gaussian_file, float sigma) {
    cout << "----- 5.4" << endl;

    vecf gaussian_lena = load(FILE_GAUSSIAN_LENA_5_3, 256*256);
    vecf blurry_gaussian_lena = load(other_gaussian_file, 256*256);
    float blurry_gaussian_mse = mse(blurry_gaussian_lena, gaussian_lena, 256);
    cout << "MSE between gaussian lena and other gaussian lena with sigma = " << sigma <<
                ": " << blurry_gaussian_mse << endl;
}

void session2() {
    // =========================================================================================
    // ====================================== SESSION 2 ========================================
    // =========================================================================================

    cout << "SESSION 2" << endl;

    ex4_1();

    ex4_2();

    ex4_4();

    ex5_1();

    ex5_2();

    ex5_3();

    // [SESSION 2 ex 5.4]
    ex5_4(FILE_GAUSSIAN_LENA_5_4_1, 1);
    ex5_4(FILE_GAUSSIAN_LENA_5_4_1_5, 1.5);

}

void ex6_1() {
    cout << "----- 6.1" << endl;
    vecf dct_mat = createDCTmatrix(256);
    store(FILE_DCT_MATRIX, dct_mat);

}

void ex6_2() {
    cout << "----- 6.2" << endl;
    vecf dct_mat = createDCTmatrix(256);
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
    vecf dct = createDCTmatrix(256);
    vecf idct = createIDCTmatrix(256);
    store(FILE_IDCT_MATRIX, idct);
    bool iso = isOrthonormal(idct, dct, 256);
    cout << "is orthogonal ? " << iso << endl;
}

void ex7_1() {
    cout << "----- 7.1" << endl;
    vecf original_lena = load(FILE_LENA, 256*256);
    vecf res = transform(original_lena, 256);
    store(FILE_DCT_LENA, res);
}

void ex7_2() {
    cout << "----- 7.2" << endl;
    vecf dct_lena = load(FILE_DCT_LENA, 256*256);
    vecf threshold_lena = threshold(dct_lena, 10, 256);
    store(FILE_THRESHOLD_DCT_LENA, threshold_lena);
    cout << "MSE DCT lena - threshold DCT lena = " << mse(dct_lena, threshold_lena, 256) << endl;
    cout << "PSNR DCT lena - threshold DCT lena = " << psnr(dct_lena, threshold_lena,255, 256) << endl;
}

void ex7_3_1() {
    cout << "----- 7.3.1" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    vecf transformed = transform(lena, 256);
    vecf reconstructed = inverseTransform(transformed, 256);
    store(FILE_RECONSTRUCTED_DCT_LENA, reconstructed);
}

void ex7_3_thresh(string filename, float t) {
    vecf lena = load(FILE_LENA, 256*256);
    vecf transformed = transform(lena, 256);
    vecf transformed_thresh = threshold(transformed, t, 256);
    vecf reconstructed = inverseTransform(transformed_thresh, 256);
    float psnr_val = psnr(reconstructed, lena, 255, 256);
    cout << "PSNR reconstructed lena with threshold " << t << " : " << psnr_val << endl;
    store(filename, reconstructed);
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
    vecf Qtable = getQtable();
    store(FILE_QUANTIZATION, Qtable);
}

void ex8_2() {
    cout << "----- 8.2" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    vecf Qtable = getQtable();
    vecf decoded = approximate(lena, FILE_DCT_8_2, FILE_QDCT_8_2, FILE_IQDCT_8_2, FILE_IQIDCT_8_2, Qtable, 256, 8);
    cout << "MSE approximated lena - lena = " << mse(lena, decoded, 256) << endl;
    cout << "PSNR approximated lena - lena = " << psnr(lena, decoded, 255, 256) << endl;
}

void ex8_3() {
    cout << "----- 8.3" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    clip2(FILE_8BPP_LENA, lena, 256);
}

void ex8_4() {
    cout << "----- 8.4" << endl;
    vecf approximated_lena = load(FILE_QDCT_8_2, 256*256);
    clip2(FILE_QDCT_8_2_8bpp, approximated_lena, 256);
    vecf decoded_approximated_lena = load(FILE_IQIDCT_8_2, 256*256);
    clip2(FILE_IQIDCT_8_2_8bpp, decoded_approximated_lena, 256);
}

void ex8_5() {
    cout << "----- 8.5" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    vecf Qtable = getQtable();

    vecf encoded_lena = encode(lena, Qtable, 256, 8);
    store(FILE_ENCODED_LENA8_5, encoded_lena);
    vecf decoded_lena = decode(encoded_lena, Qtable, 256, 8);
    store(FILE_DECODED_LENA8_5, decoded_lena);
}

void ex9_1_1() {
    cout << "----- 9.1.1" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    vecf Qtable = getQtable();

    vecf contiguous_dct_lena = encode(lena, Qtable, 256, 8);
    store(FILE_CONTIGUOUS_9, contiguous_dct_lena);
}

void ex9_1_2() {
    cout << "----- 9.1.2" << endl;
    // transformm a 256x256 pixels layout with 32x32 grid of 8x8 contiguous DCT coefs :
    vecf contiguous_dct_lena = load(FILE_CONTIGUOUS_9, 256*256);
    vecf interleaved_dct_lena = interleavedLayout(contiguous_dct_lena, 8, 256);
    store(FILE_INTERLEAVED_9, interleaved_dct_lena);

    // inverse interleave :
    vecf inv_interleaved = interleavedLayout(interleaved_dct_lena, 32, 256);
    store(FILE_INV_INTERLEAVED_9, inv_interleaved);
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
    vecf quantized_DC = load(FILE_QDCT_8_2, 256*256);
    vecf res = quantizedDCtermsMat(quantized_DC, 8, 256);
    store(FILE_32_QDCT, res);
    vecf downsized_lena = load(FILE_AVERAGE_DOWNSIZED, 32*32);
    cout << "MSE quantized DC lena - downsized average lena = " << mse(quantized_DC, downsized_lena, 32) << endl;
    cout << "PSNR quantized DC lena - downsized average lena = " << psnr(quantized_DC, downsized_lena, 255,32) << endl;
}

void ex10_2() {
    cout << "----- 10.2" << endl;
    vecf QDCTmat = load(FILE_32_QDCT, 32*32);
    vecf deltaQDCTmat = deltaEncoding(QDCTmat, 32);
    storeTXTmatrix(FILE_DELTA_DC_TXT, deltaQDCTmat, 32*32);
}

void ex10_3() {
    cout << "----- 10.3" << endl;
    vecf delta_QDCterms = loadTXTmatrix(FILE_DELTA_DC_TXT, 32*32);
    vecf deltaDecodedQDCT = deltaDecoding(delta_QDCterms, 32);
    store(FILE_RECONSTRUCTED_DELTA, deltaDecodedQDCT);
}

void ex11_1() {
    cout << "----- 11.1" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    vecf Qtable = getQtable();
    vecf contiguous_qdct_lena = encode(lena, Qtable, 256, 8);

    vecf interleaved_qdct_lena = interleavedLayout(contiguous_qdct_lena, 8, 256);
    veci zigzag_pattern = zigzagPattern(256);
    veci DC_indices = getDCIndicesFromInterleaved(32, 256);
    cout << "nb of DC : " << DC_indices.size() << endl;
    veci AC_pattern = getACzigzagPattern(getDCIndicesFromInterleaved(32, 256), zigzagPattern(256), 256);
    veci DC_pattern = getDCzigzagPattern(getDCIndicesFromInterleaved(32, 256), zigzagPattern(256), 256);
    cout << "nb of AC : " << AC_pattern.size() << endl;
}

vecf2 ex11_2() {
    cout << "----- 11.2" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    // showRunLengths(encodedRLE);
    storeRLE(FILE_11_2_TXT, encodedRLE);
    return encodedRLE;
}

void ex11_3() {
    cout << "----- 11.3" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    vecf deltaEncodedDC = loadTXTmatrix(FILE_DELTA_DC_TXT, 32*32);

    vecf decoded = decodeRLE(encodedRLE, deltaEncodedDC, 256, 8);
    store(FILE_DECODED_11_3, decoded);
}

void ex12_1() {
    cout << "----- 12.1" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    vecf P = occurences(encodedRLE);
    int N = P.size();
    storeTXTmatrix(FILE_12_2_OCC, P, N);
    cout << "longuest run length N produced : " << P.size()-1 << endl;
    cout << "nb of runs M necessary :" << encodedRLE.size() << endl;
}

void ex12_2() {
    cout << "----- 12.2" << endl;
    vecf lena = load(FILE_LENA, 256*256);
    img32 encodedDC;
    vecf2 encodedRLE = encodeRLE(lena, encodedDC, 256, 8);
    vecf P = occurences(encodedRLE);
    vecf normalized_encoded = normalizeOccurences(P);
    int N = normalized_encoded.size();
    storeTXTmatrix(FILE_12_2_NORMALIZED, normalized_encoded, N);
}

void ex12_3() {
    cout << "----- 12.3" << endl;
    vecf lena = load(FILE_LENA, 256*256);
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

    vecf lena = load(FILE_LENA, 256*256);
    float deltaEncodedDC[(256/8)*(256/8)];
    compress(FILE_BITSTREAM_DC_14_1, FILE_BITSTREAM_AC_14_1, deltaEncodedDC, lena, 256, 8);
}

void ex14_2() {
    cout << "----- 14.2" << endl;
    string compressed_bitstream = loadBitstream(FILE_BITSTREAM_AC_14_1);
    vecf deltaEncodedDC = loadTXTmatrix(FILE_DELTA_DC_14_1, 32*32);
    string bitstream_DC = loadBitstream(FILE_BITSTREAM_DC_14_1);
    vecf deltaEncodedDC2 = bitstreamDCToDelta(bitstream_DC);
//    assert((std::equal(std::begin(deltaEncodedDC), std::end(deltaEncodedDC), std::begin(deltaEncodedDC2)) == true));

    vecf decompressed_lena = decompress(FILE_BITSTREAM_DC_14_1, FILE_BITSTREAM_AC_14_1, 256, 8);
    store(FILE_DECOMPRESSED_LENA_14_2, decompressed_lena);
    clip2(FILE_CLIP_14_2, decompressed_lena, 256);

    vecf original_lena = load(FILE_LENA, 256*256);
    cout << "MSE original lena - decompressed lena = " << mse(original_lena, decompressed_lena, 256) << endl;
    cout << "PSNR original lena - decompressed lena = " << psnr(original_lena, decompressed_lena, 255, 256) << endl;
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
