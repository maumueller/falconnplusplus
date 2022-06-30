#include "BF.h"
#include "Utilities.h"
#include "Header.h"

#include <string>

/**
Retrieve topK MIPS entries using brute force computation

Input:
- MatrixXd: MATRIX_X (col-major) of size D X N
- MatrixXd: MATRIX_Q (col-major) of size D x Q

**/
void BF_TopK_MIPS()
{
    auto start = chrono::high_resolution_clock::now();

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q); // K x Q

    // For each query
    #pragma omp parallel for
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        // Reset
        priority_queue<IFPair, vector<IFPair>, greater<IFPair>> queTopK;

        // Get query
        VectorXf vecQuery = MATRIX_Q.col(q); // D x 1

        // Compute vector-matrix multiplication
        VectorXf vecRes = vecQuery.transpose() * MATRIX_X; // (1 x D) * (D x N)

        // cout << vecRes.maxCoeff() << endl;

        // Insert into priority queue to get TopK
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            float fValue = vecRes(n);

            //cout << dValue << endl;

            // If we do not have enough top K point, insert
            if (int(queTopK.size()) < PARAM_MIPS_TOP_K)
                queTopK.push(IFPair(n, fValue));
            else // we have enough,
            {
                if (fValue > queTopK.top().m_fValue)
                {
                    queTopK.pop();
                    queTopK.push(IFPair(n, fValue));
                }
            }
        }

        // Save result into mat_topK
        // Note that priorityQueue pop smallest element firest

        IVector vecTopK(PARAM_MIPS_TOP_K, -1);
        for (int k = PARAM_MIPS_TOP_K - 1; k >= 0; --k)
        {
            vecTopK[k] = queTopK.top().m_iIndex;
            queTopK.pop();
        }

        matTopK.col(q) = Map<VectorXi>(vecTopK.data(), PARAM_MIPS_TOP_K);

    }

    auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Querying Time: " << (float)duration.count() << " ms" << endl;

    //cout << matTopK << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "BF_TopK_" + int2str(PARAM_MIPS_TOP_K) + ".txt";

        outputFile(matTopK, sFileName);
    }
}
