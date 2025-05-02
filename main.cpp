#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <functional>
#include <random>
#include <pthread.h>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/global_control.h>
using namespace std;
using namespace tbb;
typedef vector<vector<double>> Matrix;
void GenerateMatrix(Matrix& mat, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);
    mat.resize(size, vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            mat[i][j] = dis(gen);
}
void DGEMM(const Matrix& A, const Matrix& B, Matrix& C, int sizeA, int sizeB) {
    for (int i = 0; i < sizeA; i++)
        for (int j = 0; j < sizeB; j++)
            for (int k = 0; k < sizeA; k++)
                C[i][j] += A[i][k] * B[k][j];
}
void DGEMM_opt_1(const Matrix& A, const Matrix& B, Matrix& C, int sizeA, int sizeB) {
    Matrix B_T(sizeB, vector<double>(sizeA));
    for (int i = 0; i < sizeA; i++)
        for (int j = 0; j < sizeB; j++)
            B_T[j][i] = B[i][j]; 

    for (int i = 0; i < sizeA; i++)
        for (int j = 0; j < sizeB; j++) {
            double sum = 0.0;
            for (int k = 0; k < sizeA; k++)
                sum += A[i][k] * B_T[j][k]; 
            C[i][j] = sum;
        }
}
void DGEMM_opt_2(const Matrix& A, const Matrix& B, Matrix& C, int sizeA, int sizeB, int blockSize) {
    for (int i = 0; i < sizeA; i += blockSize) {
        for (int j = 0; j < sizeB; j += blockSize) {
            for (int k = 0; k < sizeA; k += blockSize) {
                for (int ii = i; ii < min(i + blockSize, sizeA); ii++)
                    for (int jj = j; jj < min(j + blockSize, sizeB); jj++)
                        for (int kk = k; kk < min(k + blockSize, sizeA); kk++)
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
            }
        }
    }
}
struct PThreadArgs {
    const Matrix* A;
    const Matrix* B;
    Matrix* C;
    int sizeA;
    int sizeB;
    int startRow;
    int endRow;
};
void* pthread_worker(void* arg) {
    auto* a = static_cast<PThreadArgs*>(arg);
    for (int i = a->startRow; i < a->endRow; ++i)
        for (int j = 0; j < a->sizeB; ++j)
            for (int k = 0; k < a->sizeA; ++k)
                (*a->C)[i][j] += (*a->A)[i][k] * (*a->B)[k][j];
    return nullptr;
}
void DGEMM_POSIX(const Matrix& A, const Matrix& B, Matrix& C, int sizeA, int sizeB, int numThreads) {
    vector<pthread_t> threads(numThreads);
    vector<PThreadArgs> args(numThreads);
    int chunk = sizeA / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int start = t * chunk;
        int end   = (t == numThreads - 1) ? sizeA : start + chunk;
        args[t] = { &A, &B, &C, sizeA, sizeB, start, end };
        pthread_create(&threads[t], nullptr, pthread_worker, &args[t]);
    }
    for (auto& th : threads)
        pthread_join(th, nullptr);
}
void DGEMM_OpenMP(const Matrix& A, const Matrix& B, Matrix& C, int sizeA, int sizeB) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < sizeA; i++)
        for (int j = 0; j < sizeB; j++)
            for (int k = 0; k < sizeA; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void DGEMM_Intel_TTB(const Matrix& A, const Matrix& B, Matrix& C, int sizeA, int sizeB) {
    parallel_for(blocked_range2d<int>(0, sizeA, 0, sizeB), [&](const blocked_range2d<int>& r) {
        for (int i = r.rows().begin(); i < r.rows().end(); ++i)
            for (int j = r.cols().begin(); j < r.cols().end(); ++j)
                for (int k = 0; k < sizeA; ++k)
                    C[i][j] += A[i][k] * B[k][j];
    });
}

void RunANDMeter(function<void()> func, const string& label, ofstream& outFile, int sizeA, int sizeB) {
    auto start = chrono::high_resolution_clock::now();
    func();  
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    outFile << label << ";" << duration.count() << ";" << sizeA << "*" << sizeB << endl;
    cout << label << " Время: " << duration.count() << " секунд для " << sizeA << "x" << sizeB << endl;
}

int main() {
    int N;
    cout << "Введите размерность матрицы ==>> ";
    cin >> N;
    vector<int> threadCounts = {1, 2, 4, 8, 16, 32};
    int blockSize = 64;
    Matrix A, B, C;
    GenerateMatrix(A, N);
    GenerateMatrix(B, N);

    ofstream outFile("times.csv");
    for (int t : threadCounts) {
        //POSIX Threads
        C.assign(N, vector<double>(N, 0.0));
        RunANDMeter([&]() {
            DGEMM_POSIX(A, B, C, N, N, t);
        }, "POSIX", outFile, t, N);

        //OpenMP
        omp_set_num_threads(t);
        C.assign(N, vector<double>(N, 0.0));
        RunANDMeter([&]() {
            DGEMM_OpenMP(A, B, C, N, N);
        }, "OpenMP", outFile, t, N);

        //Intel TBB
        {
            tbb::global_control ctl(tbb::global_control::max_allowed_parallelism, t);
            C.assign(N, vector<double>(N, 0.0));
            RunANDMeter([&]() {
                DGEMM_Intel_TTB(A, B, C, N, N);
            }, "Intel_TTB", outFile, t, N);
        }
        if(t == 1) {
            outFile << endl << endl;
        }
    }

    outFile.close();
    return 0;
}