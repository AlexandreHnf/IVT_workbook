#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <assert.h>
#define PI 3.14

using namespace std;

typedef float img[];

typedef vector<float> vecf;
typedef vector<vector<float>> vecf2;
typedef vector<int> veci;


float I(int x, int y) {
    return 0.5 + 0.5*cos(y*PI/32.0) * cos(x*PI/64.0);
}

int r(int i, int j, int N) {
    // return index corresponding to raster order in a N*N matrix
    return i*N+j;
}

vecf generateCosPatternImage(int N) {
    // Generate a 256x256 pixels image with cosine pattern SESSION 1 : ex 2.1
    vecf image(N*N);
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++){
            image[r(x,y,N)] = I(x,y);
        }
    }
    return image;
}

vecf generateUDRN(float range_low, float range_high, int N) {
    // Generate a 256x256 pixels image with uniform-distributed random numbers
    vecf image(N*N);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> uniform_distribution(range_low, range_high);
    for (int x=0; x<N; ++x) {
        for (int y=0; y<N; y++) {
            // Use distribution to transform the random unsigned int generated by gen into a
            // double in [1, 2). Each call to dis(gen) generates a new random double
            image[r(x,y,N)] = uniform_distribution(generator);
        }
    }
    return image;
}

vecf generateGDRN(float mean, float stddev, int N) {
    // Generate a 256x256 pixels image with Gaussian-distributed random numbers [SESSION 2 : 4.2]
    vecf image(N*N);
    std::default_random_engine generator;
    std::normal_distribution<float> normal_distribution(mean, stddev);
    for (int x=0; x<N; ++x) {
        for (int y=0; y<N; y++) {
            image[r(x,y,N)] = normal_distribution(generator);
        }
    }
    return image;
}


void displayImage(vecf image, string name, int N) {
    cout << name << endl;
    for (int x=0; x < N; x++ ){
        for (int y=0; y < N; y++) {
            cout << image[r(x,y,N)] << " ";
        }
        cout << endl;
    }
}

void store(string filename, vecf image) {
    ofstream file;
    file.open(filename, ios::out | ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&image[0]), image.size()*sizeof(float));
    } else {cout << "Error while opening the file : " << filename << endl; }
    file.close();
}

vecf load(string filename, int N) {
    ifstream file;
    file.open(filename, ios::in|ios::binary);
    vecf image(N);
    if (!file.is_open()) {
        cout << "Error while opening the file : " << filename << endl;
    } else {
        file.seekg(0);
        file.read((char *) &image[0], image.size()*sizeof(float));
    }
    file.close();
    return image;
}

void TEST_store_load(string test_filename) {
    vecf A {1,2,3,4,5,6,7,8,9};
    store(test_filename, A);
    vecf B = load(test_filename, 9);
    assert (std::equal(std::begin(B), std::end(B), std::begin(A)) == true);

}

vecf imageProduct(vecf i1, vecf i2, int N) {
    // multiply 2 matrices pixel by pixel
    vecf i3(N*N);
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            i3[r(x,y,N)] = i1[r(x,y,N)] * i2[r(x,y,N)];
        }
    }
    return i3;
}

void TEST_imageProduct() {
    vecf A{1,2,3,4};
    vecf B{1,2,3,4};
    vecf res{1,4,9,16};
    vecf C = imageProduct(A, B, 2);
    assert (std::equal(std::begin(res), std::end(res), std::begin(C)) == true);
}

vecf imageAddition(vecf i1, vecf i2, int N) {
    // add 2 images pixel by pixel
    vecf res(N*N);
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            res[r(x,y,N)] = i1[r(x,y,N)] + i2[r(x,y,N)];
        }
    }
    return res;
}

void TEST_imageAddition() {
    vecf A{1,2,3,4};
    vecf B{1,2,3,4};
    vecf res{2,4,6,8};
    vecf C = imageAddition(A, B, 2);
    assert (std::equal(std::begin(res), std::end(res), std::begin(C)) == true);
}

float mse(vecf i1, vecf i2, int N) {
    // compute the mean squared error between 2 images // SESSION 1 : ex 3.4
    float mse = 0;
    for (int x=0; x<N; x++ ) {
        for (int y=0; y<N; y++) {
            mse += pow(i1[r(x,y,N)] - i2[r(x,y,N)], 2);
        }
    }
    return mse / (N*N);
}

void TEST_mse() {
    vecf A{1,2,3,4};
    vecf B{0.5,1,1.5,3};
    float m = mse(A, B, 2);
    assert(m == 1.125);
}

float psnr(vecf i1, vecf i2, int d, int N) {
    // compute the PSNR between two images given a MAX value // SESSION 1 : ex 3.5
    return 10.0*log10( pow(d, 2) / mse(i1, i2, N) );
}

void TEST_psnr() {
    vecf A{1,2,3,4};
    vecf B{0.5,1,1.5,4};
    float p = psnr(A, B, 255, 2);
    assert(abs(p - 48.7107) <= 0.001);
}

vecf transposeSquareMatrix(vecf m, int N) {
    vecf t(N*N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            t[r(j,i,N)] = m[r(i,j,N)]; // transpose[j][i] = m[i][j];
        }
    }
    return t;
}

void TEST_transposeSquareMatrix() {
    vecf A{1,2,3,4,5,6,7,8,9};
    vecf T{1,4,7,2,5,8,3,6,9};
    vecf B = transposeSquareMatrix(A, 3);
    assert((std::equal(std::begin(B), std::end(B), std::begin(T)) == true));
}

vecf squareMatrixMultiplication(vecf a, vecf b, int N) {
    vecf mul(N*N);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            mul[r(i,j,N)] = 0; // mul[i][j]=0;
            for(int k=0; k<N; k++){
                mul[r(i,j,N)] += a[r(i,k,N)] * b[r(k,j,N)];      //mul[i][j]+=a[i][k]*b[k][j];
            }
        }
    }
    return mul;
}

void TEST_squareMatrixMultiplication() {
    vecf a{2,4,1,2,3,9,3,1,8};
    vecf b{1,2,3,3,6,1,2,4,7};
    vecf c{16,32,17,29,58,72,22,44,66};
    vecf m = squareMatrixMultiplication(a,b, 3);
    assert((std::equal(std::begin(m), std::end(m), std::begin(c)) == true));
}

bool isIdentityMatrix(vecf A, int N) {
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            float val = round(A[r(i,j,N)]);
            if ((i != j and val != 0) || (i == j and val != 1)) {
                return false;
            }
        }
    }
    return true;
}

void TEST_isIdentityMatrix() {
    vecf c{1,0,0,0,1,0,0,0,1};
    assert(1 == isIdentityMatrix(c, 3));
}

bool isOrthonormal(vecf A, vecf B, int N) {
    vecf I = squareMatrixMultiplication(A, B, N); // AAt =?= I
    return isIdentityMatrix(I, N);
}

void TEST_isOrthonormal() {
    vecf A{0,1,0,0,0,1,1,0,0};
    vecf B{0,0,1,1,0,0,0,1,0};
    assert(isOrthonormal(A, B, 3) == true);
}

vecf createDCTmatrix(int N) {
    vecf dct(N*N);
    float ortho_rescaling = sqrt(2.0/N); // to have an orthonormal DCT-II matrix
    for (int k = 0; k < N; k++) { // lines
        for (int n = 0; n < N; n++) { // columns
            dct[r(k,n,N)] = ortho_rescaling*cos( ((M_PI*k)/(2.0*N)) * (2.0*n+1));
            if (k == 0) {dct[r(k,n,N)] = dct[r(k,n,N)]*(1.0/sqrt(2));}
        }
    }
    return dct;
}

vecf createIDCTmatrix(int N) {
    vecf dct = createDCTmatrix(N);
    vecf idct = transposeSquareMatrix(dct, N);
    return idct;
}

vecf transform(vecf X, int N) {
    vecf A = createDCTmatrix(N);
    vecf Xt = transposeSquareMatrix(X, N);
    vecf AXt = squareMatrixMultiplication(A, Xt, N);
    vecf AXtt = transposeSquareMatrix(AXt, N);
    vecf res = squareMatrixMultiplication(A, AXtt, N);
    return res;
}

vecf inverseTransform(vecf T, int N) {
    vecf A = createDCTmatrix(N);
    vecf At = transposeSquareMatrix(A, N);
    vecf AXtt = squareMatrixMultiplication(At, T, N); // AXtt
//    float Xt[N*N];
    vecf reconstructed = squareMatrixMultiplication(AXtt, A, N);
    return reconstructed;
}

vecf threshold(vecf image, float t, int N) {
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            float abs_value = abs(image[r(i,j,N)]);
            if (abs_value < t) {
                image[r(i,j,N)] = 0;
            }
        }
    }
    return image;
}

vecf getQtable() {
    vecf Qtable {16,11,10,16,24,40,51,61,
                 12,12,14,19,26,58,60,55,
                 14,13,16,24,40,57,69,56,
                 14,17,22,29,51,87,80,62,
                 18,22,37,56,68,109,103,77,
                 24,35,55,64,81,104,113,92,
                 49,64,78,87,103,121,120,101,
                 72,92,95,98,112,100,103,99};
    return Qtable;
}

vecf quantization(vecf Qtable, vecf dct_coeffs, int N) {
    vecf quantized(N*N);
    for (int i=0; i<N; i++) {
        for (int j = 0; j < N; j++) {
            float B = round(dct_coeffs[r(i,j,N)] / Qtable[r(i,j,N)]);
            quantized[r(i,j,N)] = B;
        }
    }
    return quantized;
}

vecf inverseQuantization(vecf Qtable, vecf Qcoeffs, int N) {
    vecf unquantized(N*N);
    for (int i=0; i<N; i++) {
        for (int j = 0; j < N; j++) {
            float B = Qcoeffs[r(i,j,N)] * Qtable[r(i,j,N)];
            unquantized[r(i,j,N)] = B;
        }
    }
    return unquantized;
}

vecf approximate(vecf A, string f_A_dct, string f_A_Qdct, string f_A_IQdct, string f_A_IQidct, vecf Qtable, int N, int b) {
    // DCT transform + Quantization + inverse quantization + inverse DCT transform
    // b = blocksize
    vecf dct1D = createDCTmatrix(b);
    vecf idct = transposeSquareMatrix(dct1D, b);
    vecf A_dct(N*N); vecf A_Qdct(N*N); vecf A_IQdct(N*N); vecf A_IQidct(N*N);

    for (int i=0; i<N/b; i++) {
        for (int j=0; j<N/b; j++) {
            vecf block(N*N);
            veci block_indices(b*b);  // all indices of a block w.r.t to the N*N image
            for (int k=0; k<b*b; k++) {
                int block_index = i*N*b + j*b + (k/b)*N + k%b;
                block_indices[k] = block_index;
                block[k] = A[block_index];
            }
            // end of block
            vecf transformed_block = transform(block, b);
            for (int l=0; l < b*b; l++) {A_dct[block_indices[l]] = transformed_block[l];}
            vecf Qdct_block = quantization(Qtable, transformed_block, b);
            for (int l=0; l < b*b; l++) {A_Qdct[block_indices[l]] = Qdct_block[l];}
            vecf IQdct_block = inverseQuantization(Qtable, Qdct_block, b);
            for (int l=0; l < b*b; l++) {A_IQdct[block_indices[l]] = IQdct_block[l];}
            vecf IQidct_block = inverseTransform(IQdct_block, b);
            for (int l=0; l < b*b; l++) {A_IQidct[block_indices[l]] = IQidct_block[l];}
        }
    }
    store(f_A_dct, A_dct); store(f_A_Qdct, A_Qdct); store(f_A_IQdct, A_IQdct); store(f_A_IQidct, A_IQidct);
    return A_IQidct;
}

vecf encode(vecf A, vecf Qtable, int N, int b) {
    // DCT transform + Quantization
    // b = blocksize
    vecf dct1D = createDCTmatrix(b); // DCT matrix
    vecf idct = transposeSquareMatrix(dct1D, b); // IDCT matrix
    vecf encoded(N*N);

    for (int i=0; i<N/b; i++) {
        for (int j = 0; j < N / b; j++) {
            vecf block(N*N);
            veci block_indices(b*b);  // all indices of a block w.r.t to the N*N image
            for (int k = 0; k < b * b; k++) {
                int block_index = i * N * b + j * b + (k / b) * N + k % b;
                block_indices[k] = block_index;
                block[k] = A[block_index];
            }
            // end of block
            vecf transformed_block = transform(block, b);
            vecf Qdct_block = quantization(Qtable, transformed_block, b);
            for (int l = 0; l < b * b; l++) { encoded[block_indices[l]] = Qdct_block[l]; }
        }
    }
    return encoded;
}

vecf decode(vecf encoded, vecf Qtable, int N, int b) {
    // Decode the quantized DCT coefficients by applying an inverse transform
    // b = blocksize
    vecf decoded(N*N);
    for (int i=0; i<N/b; i++) {
        for (int j = 0; j < N / b; j++) {
            vecf block(N*N);
            veci block_indices(b*b);  // all indices of a block w.r.t to the N*N image
            for (int k = 0; k < b * b; k++) {
                int block_index = i * N * b + j * b + (k / b) * N + k % b;
                block_indices[k] = block_index;
                block[k] = encoded[block_index];
            }
            // end of block
            vecf IQdct_block = inverseQuantization(Qtable, block, b);
            vecf IQidct_block = inverseTransform(IQdct_block, b);
            for (int l=0; l < b*b; l++) {decoded[block_indices[l]] = IQidct_block[l];}
        }
    }
    return decoded;
}

//void clip(string filename, img image, int N) {
//    // generate a grayscale 8x8 pixels image
//    uint8_t grayscale_8bpp[N*N];
//    for (int i=0; i<N; i++){
//        for (int j=0; j<N; j++) {
//            float value = round(image[r(i,j,N)]);
//            if (value < 0.0) {
//                value = 0.0;
//            } else if (value > 256) {
//                value = 256;
//            }
//            grayscale_8bpp[r(i,j,N)] = static_cast<uint8_t>(value);
//        }
//    }
//
//    ofstream file;
//    file.open(filename, ios::out | ios::binary);
//    if (file.is_open()) {
//        file.write(reinterpret_cast<const char *>(grayscale_8bpp), sizeof(uint8_t[N*N]));
//    } else {cout << "Error while opening the file : " << filename << endl; }
//    file.close();
//}

void clip2(string filename, vecf image, int N) {
    // generate a grayscale 8x8 pixels image
    vector<uint8_t> grayscale_8bpp(N*N);
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++) {
            float value = round(image[r(i,j,N)]);
            if (value < 0.0) {
                value = 0.0;
            } else if (value > 256) {
                value = 256;
            }
            grayscale_8bpp[r(i,j,N)] = static_cast<uint8_t>(value);
        }
    }

    ofstream file;
    file.open(filename, ios::out | ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&grayscale_8bpp[0]), grayscale_8bpp.size()*sizeof(uint8_t));
    } else {cout << "Error while opening the file : " << filename << endl; }
    file.close();
}

vecf interleavedLayout(vecf contiguous_layout, int b, int N) {
    vecf interleaved(N*N);
    int Nb = N/b;
    for (int i = 0; i < Nb; i++) {
        for (int j = 0; j < Nb; j++) {
            for (int l = 0; l < b; l++) {
                for (int m = 0; m < b; m++) {
                    int k = l*b+m;
                    int bi = i * N * b + j * b + (k / b) * N + k % b;

                    int n = i*Nb+j;
                    int li = l*N*Nb + m*Nb + (n / Nb)*N + n%Nb;

                    interleaved[li] = contiguous_layout[bi];
                }
            }
        }
    }
    return interleaved;
}

void TEST_interleavedLayout() {
    vecf A{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vecf I{0,2,1,3,8,10,9,11,4,6,5,7,12,14,13,15};
    vecf B = interleavedLayout(A, 2, 4);
    assert((std::equal(std::begin(B), std::end(B), std::begin(I)) == true));
}

vecf quantizedDCtermsMat(vecf image, int b, int N) {
    // get the quantized DC coefficients matrix of N/b * N/b elements
    vecf res(N/b * N/b);
    int Nb = N / b;
    for (int i = 0; i < Nb; i++) {
        for (int j = 0; j < Nb; j++) {
            int DC_index = i * N * b + j * b;
            res[r(i,j,Nb)] = image[DC_index];
        }
    }
    return res;
}

vecf deltaEncoding(vecf image, int N) {
    // successive differences between terms
    // [2, 4, 6, 9, 7] ====>  [2, 2, 2, 3, −2]
    vecf res(N*N);
    for (int i = N*N-1; i>0; i--) {
        res[i] = image[i] - image[i-1];
    }
    res[0] = image[0];
    return res;
}

void TEST_deltaEncoding() {
    vecf A{2,4,6,9,7,8,2,3,1};
    vecf res{2,2,2,3,-2,1,-6,1,-2};
    vecf B = deltaEncoding(A, 3);
    assert((std::equal(std::begin(res), std::end(res), std::begin(B)) == true));
}

vecf deltaDecoding(vecf image, int N) {
    // go back from delta encoding
    // [2, 2, 2, 3, −2] ====> [2, 4, 6, 9, 7]
    vecf res(N*N);
    res[0] = image[0];
    for (int i = 1; i < N*N; i++) {
        res[i] = image[i] + res[i-1];
    }
    return res;
}

void TEST_deltaDecoding() {
    vecf A{2,2,2,3,-2,1,-6,1,-2};
    vecf res{2,4,6,9,7,8,2,3,1};
    vecf B = deltaDecoding(A, 3);
    assert((std::equal(std::begin(res), std::end(res), std::begin(B)) == true));
}

void storeTXTmatrix(string filename, vecf matrix, int N) {
    ofstream myfile (filename);
    if (myfile.is_open()) {
        for(int i = 0; i < N; i ++){
            myfile << matrix[i] << " " ;
        }
        myfile.close();
    }
//    else cout << "Unable to open file";
}

vecf loadTXTmatrix(basic_string<char> filename, int N) {
    vecf res(N);
    string line;
    ifstream myfile (filename);
    if (myfile.is_open()) {
        getline(myfile, line);
        stringstream ssin(line);
        int i = 0;
        while (ssin.good() && i < N) {
            ssin >> res[i];
            ++i;
        }
        myfile.close();
    }
    return res;

//    else cout << "Unable to open file";
}

void TEST_store_load_txt(string filename) {
    vecf A{1,2,3,4,5,6,7,8,9};
    storeTXTmatrix(filename, A, 9);
    vecf B = loadTXTmatrix(filename, 9);
    assert((std::equal(std::begin(B), std::end(B), std::begin(A)) == true));
}

veci zigzagPattern(int N) {
    // get the indices of the zigzag pattern of an N*N image
    veci seq(N*N);
    for (int i=0; i<N*N; i++){seq[i] = i;}
    veci res;
    vector<vector<float>> sol(N+N-1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = i + j;
            if (sum % 2 == 0) {
                sol[sum].insert(sol[sum].begin(), seq[r(i,j,N)]);
            } else {
                sol[sum].push_back(seq[r(i,j,N)]);
            }
        }
    }
    for (unsigned int i = 0; i < sol.size(); i++) {
        for (unsigned int j = 0; j < sol[i].size(); j++) {
            res.push_back(sol[i][j]);
        }
    }
    return res;
}

void TEST_zigzagPattern() {
    veci Z = zigzagPattern(3);
    veci res {0,1,3,6,4,2,5,7,8};
    assert((std::equal(std::begin(Z), std::end(Z), std::begin(res)) == true));
}

vecf zigzag(vecf A, int N) {
    // get the zigzag pattern of an image
    vecf res(N*N);
    veci zigzag_pattern = zigzagPattern(N);
    for (int i=0; i<N*N; i++) {
        res[i] = A[zigzag_pattern[i]];
    }
    return res;
}

vecf inverseZigzag(vecf zigzag, veci zigzag_pattern, int N) {
    // invert the zigzag pattern to get back the interleaved image
    vecf res(N*N);
    for (int i=0; i<N*N; i++) {
        res[zigzag_pattern[i]] = zigzag[i];
    }
    return res;
}

veci getDCIndicesFromInterleaved(int b, int N) {
    // get the indicces of the DC coefficients from the interleaved pattern image
    veci indices;
    for (int k=0; k<b*b; k++) {
        int DC_index = 0*N*b + 0*b + (k/b)*N + k%b;
        indices.push_back(DC_index);
    }
    return indices;
}

void TEST_getDCIndicesFromInterleaved() {
    veci A = getDCIndicesFromInterleaved(2, 4);
    veci B {0,1,4,5};
    assert((std::equal(std::begin(A), std::end(A), std::begin(B)) == true));
}

veci getDCzigzagPattern(veci DC_indices, veci zigzag_indices, int N) {
    // get the indices of the DC coefficients from the zigzag pattern
    veci DC_pattern(N*N, -1); // init with -1 values
    for (int i=0; i<N*N; i++) {
        if (std::find(DC_indices.begin(), DC_indices.end(), zigzag_indices[i]) != DC_indices.end()) {
            DC_pattern[i] = zigzag_indices[i];
        }
    }
    return DC_pattern;
}

void TEST_getDCzigzagPattern() {
    veci zig {0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};
    veci DC_indices {0,1,4,5};
    veci res {0, 1, 4, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    veci D = getDCzigzagPattern(DC_indices, zig, 4);
    assert((std::equal(std::begin(D), std::end(D), std::begin(res)) == true));
}

veci getACzigzagPattern(veci DC_indices, veci zigzag_indices, int N) {
    // get the indices of the AC coefficients in zigzag pattern
    veci AC_pattern;
    for (int i=0; i<N*N; i++) {
        bool is_DC = false;
        if (std::find(DC_indices.begin(), DC_indices.end(), zigzag_indices[i]) != DC_indices.end()) {
            is_DC = true;
        }
        if (not is_DC) {
            AC_pattern.push_back(zigzag_indices[i]);
        }
    }
    return AC_pattern;
}

void TEST_getACzigzagPattern() {
    veci zig {0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};
    veci DC_indices {0,1,4,5};
    veci res {8, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};
    veci A = getACzigzagPattern(DC_indices, zig, 4);
    assert((std::equal(std::begin(A), std::end(A), std::begin(res)) == true));
}

vecf getBackACDCzigzag(vecf DC_coeffs, veci pattern, vecf AC_coeffs, int b, int N) {
    // get back the interleaved image by combining DC and AC coefficients
    vecf zigzag_pattern (pattern.size());
    veci DC_indices = getDCIndicesFromInterleaved(N/b, N);
    int a = 0; // index of AC
    for (int i=0; i<N*N; i++) {
        if (pattern[i] == -1) { // free space for an AC coefficient
            zigzag_pattern[i] = AC_coeffs[a];
            a++;
        } else {
            for (unsigned int j=0; j<DC_indices.size(); j++) {
                if (pattern[i] == DC_indices[j]) {
                    zigzag_pattern[i] = DC_coeffs[j];
                    break;
                }
            }
        }
    }
    return zigzag_pattern;
}

void TEST_getBackACDCzigzag() {
    vecf DC_c{1,2,5,6};
    veci DC {0, 1, 4, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    vecf AC_c {9,3,4,7,10,13,14,11,8,12,15,16};
    veci AC {8, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};
//    veci res {0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};
    vecf res {1, 2, 5, 9, 6, 3, 4, 7, 10, 13, 14, 11, 8, 12, 15, 16};
    vecf Z = getBackACDCzigzag(DC_c, DC, AC_c, 2, 4);
    assert((std::equal(std::begin(Z), std::end(Z), std::begin(res)) == true));
}


vecf getACFromInterleaved(vecf interleaved, int b, int N) {
    // Get the AC coefficients from the interleaved pattern image
    veci zigzag_pattern = zigzagPattern(N);
    veci DC_indices = getDCIndicesFromInterleaved(N/b, N);
    veci AC_pattern = getACzigzagPattern(getDCIndicesFromInterleaved(N/b, N), zigzagPattern(N), N);
    int nb_AC = N*N - (N/b)*(N/b);
    vecf AC_coeffs(nb_AC);
    for (int i=0; i<nb_AC; i++) {
        AC_coeffs[i] = interleaved[AC_pattern[i]];
    }
    return AC_coeffs;
}

void display_RLE(vector<vector<float>> rle) {
    for (unsigned int i = 0; i < rle.size(); i++){
        cout << "(" << rle[i][0] << "," << rle[i][1] << ")" << ",  ";
    }
}

vector<vector<float>> runLengthEncoding(vecf sequence, int N) {
    // Produce the run length encoding of a sequence of values
    // WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW
    // 12W1B12W3B24W1B14W

    vector<vector<float>> res;
    vector<float> current = {1, sequence[0]};
    int i = 1;
    while (i < N) {
        if ((i == N-1) || (sequence[i] != current[1])) {
            if (sequence[i] == current[1]) {current[0] ++;}
            res.push_back(current);
            current = {1, sequence[i]};
        } else {
            current[0]++;
        }
        i++;
    }
    // display_RLE(res);
    return res;
}

void TEST_runLengthEncoding() {
    vecf A{1,1,2,3,1,1,1,0,0,0,0,0,0,1,1,1};
    vecf2 B = runLengthEncoding(A, 16);
    vecf counts;
    vecf symbols;
    for (unsigned int i = 0; i < B.size(); i++) {
        counts.push_back(B[i][0]);
        symbols.push_back(B[i][1]);
    }
    vecf c {2,1,1,3,6,3};
    vecf s {1,2,3,1,0,1};
    assert((std::equal(std::begin(counts), std::end(counts), std::begin(c)) == true));
    assert((std::equal(std::begin(symbols), std::end(symbols), std::begin(s)) == true));
}

void storeRLE(string filename, vecf2 rle) {
    vecf to_store(rle.size()*2);
    int k=0;
    for (unsigned int i=0; i<rle.size(); i++) {
        to_store[k] = rle[i][0];
        to_store[k+1] = rle[i][1];
        k+=2;
    }
    storeTXTmatrix(filename, to_store, rle.size()*2);
}

int longuestRunLength(vector<vector<float>> runLengths) {
    // get the longuest run length
    int longuest = 0;
    for (unsigned int i = 0; i < runLengths.size(); i++) {
        if (runLengths[i][0] > longuest) {
            longuest = runLengths[i][0];
        }
    }
    return longuest;
}

vecf occurences(vecf2 run_lengths) {
    // Computes the occurences of each RLE symbols
    int longest_run_length = longuestRunLength(run_lengths);
    vector<float> P(longest_run_length+1, 0); // +1 because we have occurence '0' added

    for (unsigned int i=0; i<run_lengths.size(); i++) {
        P[run_lengths[i][0]] += 1;
    }
    return P;
}

void TEST_occurences() {
    vecf A{1,1,2,3,1,1,1,0,0,0,0,0,0,1,1,1};
    vecf2 B = runLengthEncoding(A, 16);
    vecf occ = occurences(B);
    vecf o {0,2,1,2,0,0,1};
    assert((std::equal(std::begin(occ), std::end(occ), std::begin(o)) == true));
}

vecf runLengthDecoding(vecf2 runLengths) {
    // Takes a run length encoded sequence and decode it
    vector<float> res;
    for (unsigned int i = 0; i < runLengths.size(); i++) {
        for (int j = 0; j < runLengths[i][0]; j++) {
            res.push_back(runLengths[i][1]);
        }
    }
    vecf decoded;
    for (unsigned int i = 0; i < res.size(); i++){
        decoded.push_back(res[i]);
    }
    return decoded;
}

void TEST_runLengthDecoding() {
    vecf A{1,1,2,3,1,1,1,0,0,0,0,0,0,1,1,1};
    vecf2 B = runLengthEncoding(A, 16);
    vecf C = runLengthDecoding(B);
    assert((std::equal(std::begin(C), std::end(C), std::begin(A)) == true));
}

void showRunLengths(vecf2 rle) {
    for (unsigned int i=0; i<rle.size(); i++) {
        cout << "(" << rle[i][0] <<"," << rle[i][1] << ")" << " ";
    }
    cout << endl;
}

vecf2 encodeRLE(vecf A, img deltaEncodedDC, int N, int b) {
    // takes an image and encode its DC coefficients with delta encoding and
    // run length encoding for AC coefficients
    vecf Qtable = getQtable();
    vecf encoded = encode(A, Qtable, N, b);
    vecf DC = quantizedDCtermsMat(encoded, b, N);
    vecf deltaDC = deltaEncoding(DC, N/b);
    for (unsigned int i=0; i<deltaDC.size(); i++) {deltaEncodedDC[i] = deltaDC[i];}

    vecf interleaved = interleavedLayout(encoded, b, N);
    int nb_AC = N*N - (N/b)*(N/b);
    vecf AC_coeffs = getACFromInterleaved(interleaved, b, N);
    vecf2 encodedRLE = runLengthEncoding(AC_coeffs, nb_AC);

    return encodedRLE;
}

vecf decodeRLE(vecf2 encodedRLE, vecf DC_coeffs, int N, int b) {
    // Takes a run length Encoding and decode into an image
    vecf AC_coeffs = runLengthDecoding(encodedRLE);
    vecf deltaDecodedDC = deltaDecoding(DC_coeffs, N/b); // DC coefficients after delta decoding
    veci DC_pattern = getDCzigzagPattern(getDCIndicesFromInterleaved(N/b, N), zigzagPattern(N), N);

    vecf zigzagACDC = getBackACDCzigzag(deltaDecodedDC, DC_pattern, AC_coeffs, b,N); // zigzag with AC + DC coeffs

    vecf inv_zigzag = inverseZigzag(zigzagACDC, zigzagPattern(N), N);

    vecf inv_interleaved = interleavedLayout(inv_zigzag, N/b, N); //get back contiguous
    vecf Qtable = getQtable();
    vecf decoded = decode(inv_interleaved, Qtable, N, b);
    return decoded;
}

vecf normalizeOccurences(vecf occ) {
    // Normalize values so that the sum of all values equals 1
    vecf normalized;
    float sum_values = 0.0;
    for (unsigned int i=0; i < occ.size(); i++) {
        sum_values += occ[i];
    }
    for (unsigned int i = 0; i < occ.size(); i++) {
        normalized.push_back((float) occ[i] / sum_values);
    }
    return normalized;
}

void TEST_normalizeOccurences() {
    vecf occ {1,2,3,4,5,6,7,8,9};
    vecf n = normalizeOccurences(occ);
    float sum = 0;
    for (unsigned int i=0; i<n.size(); i++)
        sum += n[i];
    assert(sum == 1);
}

float entropy(vecf occ) {
    // Comptes the entropy value of a Probability density function
    // Gives the theoretical number of bits per symbol
    float H = 0;
    for (unsigned int i = 0; i < occ.size(); i++) { // all symbols
        if (occ[i] != 0.0) {
            float p = occ[i];
            H = H + p*log(p);
        }
    }
    H = H * (-1);
    return  H;
}

void TEST_entropy() {
    vecf occ {2,5,1,3,4,6,2,3,1};
    float e = entropy(occ);
    assert(abs(e - -33.7072) <= 0.001);
}


string decToBinary(int n) {
    // function to convert decimal to binary

    // array to store binary number
    int binaryNum[32];
    string res = "";

    // counter for binary array
    int i = 0;
    while (n > 0) {
        // storing remainder in binary array
        binaryNum[i] = n % 2;
        n = n / 2;
        i++;
    }

    // binary array in reverse order
    for (int j = i - 1; j >= 0; j--) {
        int lol = binaryNum[j];
        res += to_string(lol);
    }
    return res;
}

void TEST_decToBinary() {
    assert(decToBinary(7) == "111");
    assert(decToBinary(10) == "1010");
    assert(decToBinary(12) == "1100");
}

int binaryToDec(string b) {
    unsigned long long dec = std::stoull(b, 0, 2);
    return dec;
}

void TEST_binaryToDec() {
    assert(binaryToDec("111") == 7);
    assert(binaryToDec("1010") == 10);
    assert(binaryToDec("1100") == 12);
}

string golomb(int nb) {
    // Encode a decimal number to Exponential Golomb encoding
    string res = "";
    string bin_nb = decToBinary(nb+1);
    int nb_zeros = bin_nb.length() - 1;
    for (int i = 0; i < nb_zeros; i++) {
        res += "0";
    }
    res += " ";
    res += bin_nb;

    return res;
}

void TEST_golomb() {
    assert(golomb(0) == " 1");
    assert(golomb(2) == "0 11");
    assert(golomb(3) == "00 100");
    assert(golomb(4) == "00 101");
    assert(golomb(5) == "00 110");
    assert(golomb(6) == "00 111");
    assert(golomb(7) == "000 1000");
}

int inverseGolombStream(istringstream& iss) {
    // invert golomb encoding using a string stream of 0s and 1s
    string zeros;
    string exp_gol;
    iss >> zeros;
    iss >> exp_gol;
    if (zeros.length() == 0 or exp_gol.length() == 0) {
        return 0;
    }
    int dec = binaryToDec(exp_gol) - 1;
    return dec;
}

int mapValueForGolomb(int x) {
    // Map the real decimal value that can be negative to a positive golomb decimal value
    if (x > 0) { // positive => map to odd integer
        return 2*x - 1;
    } else if (x < 0) { // negative => map to pair integer
        return (-2) * x;
    } else {
        return 0;
    }
}

void TEST_mapValueForGolomb() {
    assert(mapValueForGolomb(0) == 0);
    assert(mapValueForGolomb(1) == 1);
    assert(mapValueForGolomb(-1) == 2);
    assert(mapValueForGolomb(2) == 3);
    assert(mapValueForGolomb(-2) == 4);
    assert(mapValueForGolomb(3) == 5);
    assert(mapValueForGolomb(-3) == 6);
    assert(mapValueForGolomb(4) == 7);
    assert(mapValueForGolomb(-4) == 8);
}

int reverseMapForGolomb(int y) {
    // Map the golomb decimal number to its corresponding real decimal value
    if (y == 0) {return 0;}
    if (y % 2 == 0) { // pair
        return (-1) * (y/2);
    } else {
        return (y+1) / 2;
    }
}

void TEST_reverseMapForGolomb() {
    assert(reverseMapForGolomb(0) == 0);
    assert(reverseMapForGolomb(1) == 1);
    assert(reverseMapForGolomb(2) == -1);
    assert(reverseMapForGolomb(3) == 2);
    assert(reverseMapForGolomb(4) == -2);
    assert(reverseMapForGolomb(5) == 3);
    assert(reverseMapForGolomb(6) == -3);
    assert(reverseMapForGolomb(7) == 4);
    assert(reverseMapForGolomb(8) == -4);
}

void TEST_inverseGolombStream() {
    int x1 = mapValueForGolomb(0);
    istringstream iss (golomb(x1));
    assert(reverseMapForGolomb(inverseGolombStream(iss)) == 0);
    int x2 = mapValueForGolomb(1);
    istringstream iss2 (golomb(x2));
    assert(reverseMapForGolomb(inverseGolombStream(iss2)) == 1);
    int x3 = mapValueForGolomb(-1);
    istringstream iss3 (golomb(x3));
    assert(reverseMapForGolomb(inverseGolombStream(iss3)) == -1);
    int x4 = mapValueForGolomb(2);
    istringstream iss4 (golomb(x4));
    assert(reverseMapForGolomb(inverseGolombStream(iss4)) == 2);
    int x5 = mapValueForGolomb(6);
    istringstream iss5 (golomb(x5));
    assert(reverseMapForGolomb(inverseGolombStream(iss5)) == 6);
}

void storeBitstream(string filename, string b) {
    ofstream myfile (filename);
    if (myfile.is_open()) {
        myfile << b;
        myfile.close();
    }
    //else cout << "Unable to open file";
}

string loadBitstream(basic_string<char> filename) {

    string line;
    string res;
    ifstream myfile (filename);
    if (myfile.is_open()) {
        getline(myfile, line);
        stringstream ssin(line);
        ssin >> res;
        myfile.close();
    }
    //else cout << "Unable to open file";
    return res;
}

void TEST_store_load_bitstream(string filename) {
    string A = "01010101010101";
    storeBitstream(filename, A);
    string B = loadBitstream(filename);
    assert(B == A);
}

string generateBitStreamAC(vecf2 rle) {
    // convert a run length encoding to a bitstream
    // (2,1),  (1,2),  (1,3),  (3,1),  (6,0),  (2,1)
    // 1 1 2 3 1 1 1 0 0 0 0 0 0 1
    string bitstream = "";

    for (unsigned int i=0; i < rle.size(); i++) {
        string g1 = golomb(mapValueForGolomb(rle[i][0]));
        g1.erase(remove(g1.begin(), g1.end(), ' '), g1.end());
        string g2 = golomb(mapValueForGolomb(rle[i][1]));
        g2.erase(remove(g2.begin(), g2.end(), ' '), g2.end());

        bitstream += g1 + g2;
    }
    return bitstream;
}

void TEST_generateBitStreamAC() {
    vecf A{1,1,2,3,1,1,1,0,0,0,0,0,0,1,1,1};
    vecf2 B = runLengthEncoding(A, 16);
    string b = generateBitStreamAC(B);
    assert(b == "001000100100010001000110001100100001100100110010");
}

string generateBitStreamDC(vecf image, int N) {
    // image = DC coefficients image
    string bitstream = "";
    for (int i=0; i < N*N; i++) {
        string g = golomb(mapValueForGolomb(image[i]));
        g.erase(remove(g.begin(), g.end(), ' '), g.end());

        bitstream += g;
    }
    return bitstream;
}

string compress(string fileDC, string fileAC, img deltaEncodedDC, vecf image, int N, int b) {
    // compress an image into a bitstream (delta encoding + Golomb for DC coefficients and
    // Run length encoding + Variable Length encoding (Golomb) for AC coefficients)
    vecf2 encodedRLE = encodeRLE(image, deltaEncodedDC, N, b);
    vecf DC_coeffs((N/b)*(N/b));
    for (unsigned int i=0; i<DC_coeffs.size(); i++) {DC_coeffs[i] = deltaEncodedDC[i];}
    string bitstreamDC = generateBitStreamDC(DC_coeffs, N/b);
    storeBitstream(fileDC, bitstreamDC);
    string bitstreamAC = generateBitStreamAC(encodedRLE);
    storeBitstream(fileAC, bitstreamAC);
    return bitstreamAC;
}

vecf2 bitstreamACToRLE(string bitstream) {
    // 1 2 3 4
    // 0|10 0|11 00|100 00|101
    // 0100110010000101
    vecf2 decoded;
    int i = -1;
    int nb_zeros = 0;
    string current_golomb = "";
    int pair_index = 0;
    vecf current_pair(2, 0.0);
    while (i < (int) bitstream.length() - 1){
        i++;
        if (bitstream[i] == '0') {
            nb_zeros++;
            current_golomb += "0";
        } else {
            int j = i;
            current_golomb += " ";
            while (j < i+nb_zeros+1) {
                current_golomb += bitstream[j];
                j++;
            }
            i += nb_zeros;
            istringstream iss (current_golomb);
            int inv_gol = inverseGolombStream(iss);
            current_pair[pair_index] = (float) reverseMapForGolomb(inv_gol);
            if (pair_index == 1) {decoded.push_back(current_pair);}
            pair_index = (pair_index + 1)%2;
            current_golomb = "";
            nb_zeros = 0;
        }
    }
    return decoded;
}

void TEST_bitstreamACToRLE() {
    vecf A{1,1,2,3,1,1,1,0,0,0,0,0,0,1,1,1};
    vecf2 B = runLengthEncoding(A, 16);
    string b = "001000100100010001000110001100100001100100110010";
    vecf2 R = bitstreamACToRLE(b);
    assert((std::equal(std::begin(R), std::end(R), std::begin(B)) == true));
}

vecf bitstreamDCToDelta(string bitstream) {
    // convert a bitstream to an array representing the delta encoded DC coefficients
    vecf decoded;
    int i = -1;
    int nb_zeros = 0;
    string current_golomb = "";
    while (i < (int) bitstream.length() - 1){
        i++;
        if (bitstream[i] == '0') {
            nb_zeros++;
            current_golomb += "0";
        } else {
            int j = i;
            current_golomb += " ";
            while (j < i+nb_zeros+1) {
                current_golomb += bitstream[j];
                j++;
            }
            i += nb_zeros;
            istringstream iss (current_golomb);
            int inv_gol = inverseGolombStream(iss);
            float gol_value = (float) reverseMapForGolomb(inv_gol);
            decoded.push_back(gol_value);
            current_golomb = "";
            nb_zeros = 0;
        }
    }
    return decoded;
}

vecf decompress(string fileDC, string fileAC, int N, int b) {
    // decompress the image from 2 bitstreams of DC and AC coefficients
    string bitstreamDC = loadBitstream(fileDC);
    vecf deltaEncodedDC = bitstreamDCToDelta(bitstreamDC);
    string bitstreamAC = loadBitstream(fileAC);

    vecf2 decodedRLE = bitstreamACToRLE(bitstreamAC);
    vecf decompressed = decodeRLE(decodedRLE, deltaEncodedDC, N, b);
    return decompressed;
}





