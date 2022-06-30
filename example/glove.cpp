#include "NNDataStructure.h"

MatrixXf load_file(std::string fn, int D, int N) {
    FILE *f = fopen(fn.c_str(), "r");
    if (!f)
    {
        printf("Data file does not exist");
        exit(1);
    }

    FVector vecTempX(D * N, 0.0);

    for (int n = 0; n < N; ++n)
    {
        for (int d = 0; d < D; ++d)
        {
            fscanf(f, "%f", &vecTempX[n * D + d]);
        }
    }

    // Matrix_X is col-major
    return Map<MatrixXf>(vecTempX.data(), D, N);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " <data> <queries> <N> <D> <Q>" << std::endl;
        return 42;
    }
    int n = atoi(argv[3]);
    int d = atoi(argv[4]);
    int q = atoi(argv[5]);
    auto data = load_file(argv[1], d, n);
    auto queries = load_file(argv[2], d, q);
    auto index = FalconnPP(n, d);
    index.buildIndex(data);
    index.query(queries, 1);
    return 0;
}