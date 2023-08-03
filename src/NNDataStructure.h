#include<iostream>
#include<chrono>
#include<vector>

#include "Utilities.h"

using namespace std;

class FalconnPP { 
    public: 
        FalconnPP(int n, int d, int num_tables);
        void buildIndex(MatrixXf data);
        MatrixXi query(MatrixXf& query, int count, int probes);
    private:
        int PARAM_LSH_BUCKET_SIZE_LIMIT; // Size of bucket
        float PARAM_LSH_BUCKET_SIZE_SCALE; // Size of scale
        float PARAM_LSH_DISCARD_T; // Threshold to discard
        int PARAM_LSH_NUM_TABLE;
        int PARAM_LSH_NUM_PROJECTION;
        int PARAM_LSH_NUM_INDEX_PROBES;
        int PARAM_LSH_NUM_QUERY_PROBES;
        bool PARAM_LSH_PROBING_HEURISTIC = 1;

        bool PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE;
        int PARAM_INTERNAL_LSH_NUM_BUCKET;
        int PARAM_INTERNAL_LOG2_NUM_PROJECTION;
        int PARAM_INTERNAL_FWHT_PROJECTION; // for the case numProject < d
        int PARAM_INTERNAL_LOG2_FWHT_PROJECTION;
        int PARAM_INTERNAL_NUM_ROTATION = 3;

        int PARAM_TEST_LSH_L_RANGE;
        int PARAM_TEST_LSH_L_BASE;
        int PARAM_TEST_LSH_qPROBE_RANGE;
        int PARAM_TEST_LSH_qPROBE_BASE;
        float PARAM_TEST_LSH_SCALE_RANGE;
        float PARAM_TEST_LSH_SCALE_BASE;

        boost::dynamic_bitset<> bitHD1; // all 0's by default
        boost::dynamic_bitset<> bitHD2; // all 0's by default


        int PARAM_INTERNAL_LSH_RANGE;
        int PARAM_INTERNAL_LSH_BASE;

        int PARAM_DATA_D;
        int PARAM_DATA_N;
        int PARAM_QUERY_Q;

        int PARAM_MIPS_CANDIDATE_SIZE;
        int PARAM_MIPS_TOP_K; 

        int PARAM_NUM_ROTATION; 

        bool PARAM_INTERNAL_LIMIT_BUCKET = true;

        MatrixXf MATRIX_X;
        MatrixXf MATRIX_Q;

        //MatrixXi MATRIX_HADAMARD;
        MatrixXf MATRIX_HD1;
        MatrixXf MATRIX_HD2;

        //boost::multi_array<int, 3> VECTOR3D_FALCONN_TABLES;
        IVector VECTOR3D_FALCONN_TABLES;
        vector<IVector> VECTOR2D_FALCONN_TABLES;
        vector<vector<IFPair>> VECTOR2D_PAIR_FALCONN_TABLES;

        vector<pair<uint32_t, uint16_t>> VECTOR_PAIR_FALCONN_BUCKET_POS;

        vector<int> VECTOR_FALCONN_TABLES;
        vector<IFPair> VECTOR_PAIR_FALCONN_TABLES;

};

FalconnPP::FalconnPP(int n, int d, int num_tables) {
    PARAM_DATA_N = n;
    PARAM_DATA_D = d;

    PARAM_LSH_NUM_PROJECTION = 128;
    cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

    if (PARAM_LSH_NUM_PROJECTION < PARAM_DATA_D)
        PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
    else
        PARAM_INTERNAL_FWHT_PROJECTION = PARAM_LSH_NUM_PROJECTION;

    cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

    PARAM_LSH_NUM_TABLE = num_tables; 
    cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

    int iTemp = 1; 
    PARAM_LSH_BUCKET_SIZE_SCALE = iTemp * 1.0 / 100;
    cout << "Bucket scale: " << PARAM_LSH_BUCKET_SIZE_SCALE << endl;

    PARAM_LSH_NUM_INDEX_PROBES = 10;
    cout << "Number of iProbes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

    PARAM_TEST_LSH_qPROBE_BASE = 0;
    cout << "Number of base qProbes: " << PARAM_TEST_LSH_qPROBE_BASE << endl;

    PARAM_TEST_LSH_qPROBE_RANGE = 1000;
    cout << "Number of range qProbes: " << PARAM_TEST_LSH_qPROBE_RANGE << endl;

    PARAM_MIPS_CANDIDATE_SIZE = 0;

    if (PARAM_MIPS_CANDIDATE_SIZE == 0)
        PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;

    cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

    PARAM_LSH_PROBING_HEURISTIC = 1;
    PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;
    PARAM_INTERNAL_LSH_NUM_BUCKET = 2 * PARAM_LSH_NUM_PROJECTION;
    PARAM_INTERNAL_LOG2_NUM_PROJECTION = log2(PARAM_LSH_NUM_PROJECTION);
    PARAM_INTERNAL_LOG2_FWHT_PROJECTION = log2(PARAM_INTERNAL_FWHT_PROJECTION);

    PARAM_MIPS_TOP_K = 20;

}

void bitHD3Generator2(int p_iNumBit, boost::dynamic_bitset<>& bitHD1, boost::dynamic_bitset<>& bitHD2)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_int_distribution<uint32_t> unifDist(0, 1);

    bitHD1 = boost::dynamic_bitset<> (p_iNumBit);
    bitHD2 = boost::dynamic_bitset<> (p_iNumBit);

    // Loop col first since we use col-wise
    for (int d = 0; d < p_iNumBit; ++d)
    {
        bitHD1[d] = unifDist(generator) & 1;
        bitHD2[d] = unifDist(generator) & 1;
    }
}


void FalconnPP::buildIndex(MatrixXf data) {
//    cout << "Build Scaled FalconnCEOs structure..." << endl;
    auto start = chrono::high_resolution_clock::now();

    MATRIX_X = data;

    // PARAM_INTERNAL_FWHT_PROJECTION = 2^log{D}
    // There will be some case we use less than D random projection, so call FWHT with 2^log{D} and select top-D-up positions
    // This trick is also presented in Falconn
    bitHD3Generator2(PARAM_INTERNAL_FWHT_PROJECTION * PARAM_LSH_NUM_TABLE * PARAM_INTERNAL_NUM_ROTATION, bitHD1, bitHD2);




    float fScaleData = 0.0;
    int iLowerBound_Count = 0;

    // # bucket = (2 * D)^2 since we use 2 combined LSH
    // PARAM_INTERNAL_LSH_NUM_BUCKET = 2 * NUM_PROJECTION
    int NUM_BUCKET = PARAM_INTERNAL_LSH_NUM_BUCKET * PARAM_INTERNAL_LSH_NUM_BUCKET;

    /**
    Data structure:
    - VECTOR_FALCONN_TABLES is an 1D array, containing NUM-BUCKET vector Li,
    each vector L1 contains points sorted by <x, ri> + <x, si> (due to CEOs property)
    where ri and si are random vectors from 2 combined LSH

    - VECTOR_PAIR_FALCONN_BUCKET_POS is a vector of pair where
    pair.first is the beginning of the bucket in VECTOR_FALCONN_TABLES
    pair.second is the size of the bucket (i.e. size of Li)
    **/
    // pair.first is the bucket pos in a big 1D array, which is often large (L * n = 2^10 * 2^20 = 2^30), so uint32_t is okie
    // Note that we might need to use uint64_t in some large data set.
    // However, it demands more space for this array, which might be not cache-efficient
    // pair.second is the bucket size, which is often small, so uint16_t is more than enough

    VECTOR_PAIR_FALCONN_BUCKET_POS = vector<pair<uint32_t, uint16_t>> (PARAM_LSH_NUM_TABLE * NUM_BUCKET);

    // No need to init since it will be updated on-the-fly
    // VECTOR_FALCONN_TABLES = vector<IVector> (PARAM_LSH_NUM_TABLE * NUM_BUCKET);

    #pragma omp parallel for
	for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
	{
        int iBaseTableIdx = l * NUM_BUCKET;

        // vecBucket_MaxQue is a table, each bucket is a max priQueue
        vector< priority_queue< IFPair, vector<IFPair> > > vecBucket_MaxQue(NUM_BUCKET);

        /**
        Build a hash table for N points
        **/
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            VectorXf rotatedX1 = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
            rotatedX1.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);

            VectorXf rotatedX2 = rotatedX1;

            for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
            {
//                rotatedX1 = rotatedX1.cwiseProduct(matHD1.col(r));
//                rotatedX2 = rotatedX2.cwiseProduct(matHD2.col(r));

                for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
                {
                    rotatedX1(d) *= (2 * (int)bitHD1[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                    rotatedX2(d) *= (2 * (int)bitHD2[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                }

                fht_float(rotatedX1.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
                fht_float(rotatedX2.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
            }

            // This queue is used for finding top-k max hash values and hash index
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueProbes1;
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueProbes2;

            /**
            We use a priority queue to keep top-iProbes abs projection for each LSH functions
            **/
            for (int r = 0; r < PARAM_LSH_NUM_PROJECTION; ++r)
            {
                // 1st LSH
                int iSign = sgn(rotatedX1(r));
                float fAbsHashValue = iSign * rotatedX1(r);

                // if iSign >= 0: hash value = {0, ..., r-1}
                // if iSign < 0: hash value = {r, ..., 2r-1}
                int iHashIndex = r;
                if (iSign < 0)
                    iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION; // set bit at position log2(D)

                if ((int)minQueProbes1.size() < PARAM_LSH_NUM_INDEX_PROBES)
                    minQueProbes1.push(IFPair(iHashIndex, fAbsHashValue));

                // in case full queue
                else if (fAbsHashValue > minQueProbes1.top().m_fValue)
                {
                    minQueProbes1.pop();
                    minQueProbes1.push(IFPair(iHashIndex, fAbsHashValue));
                }

                // 2nd LSH
                iSign = sgn(rotatedX2(r));
                fAbsHashValue = iSign * rotatedX2(r);

                // if iSign >= 0: hash value = {0, ..., r-1}
                // if iSign < 0: hash value = {r, ..., 2r-1}
                iHashIndex = r;
                if (iSign < 0)
                    iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION; // set bit at position log2(D)

                if ((int)minQueProbes2.size() < PARAM_LSH_NUM_INDEX_PROBES)
                    minQueProbes2.push(IFPair(iHashIndex, fAbsHashValue));

                // in case full queue
                else if (fAbsHashValue > minQueProbes2.top().m_fValue)
                {
                    minQueProbes2.pop();
                    minQueProbes2.push(IFPair(iHashIndex, fAbsHashValue));
                }
            }

//            assert((int)minQueProbes1.size() == PARAM_LSH_NUM_INDEX_PROBES);
//            assert((int)minQueProbes2.size() == PARAM_LSH_NUM_INDEX_PROBES);

            // Convert to vector
            vector<IFPair> vec1(PARAM_LSH_NUM_INDEX_PROBES);
            vector<IFPair> vec2(PARAM_LSH_NUM_INDEX_PROBES);

            for (int p = PARAM_LSH_NUM_INDEX_PROBES - 1; p >= 0; --p)
            {
                vec1[p] = minQueProbes1.top();
                minQueProbes1.pop();

                vec2[p] = minQueProbes2.top();
                minQueProbes2.pop();
            }

            /**
            Find the top-iProbes over 2 LSH (bucketIndex, abs(projValue1) + abs(projValue2)
            **/
            priority_queue<IFPair, vector<IFPair>, greater<IFPair>> minQue;

            for (const auto& ifPair1: vec1)         //p: probing step
            {
                int iHashIndex1 = ifPair1.m_iIndex; //hash value
                float fAbsHashValue1 = ifPair1.m_fValue; //abs(projection value)

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iHashIndex2 = ifPair2.m_iIndex;
                    float fAbsSumHash = ifPair2.m_fValue + fAbsHashValue1;

                    int iBucketIndex = iHashIndex1 * PARAM_INTERNAL_LSH_NUM_BUCKET + iHashIndex2; // (totally we have 2D * 2D buckets)

                    // new pair for inserting into priQueue
                    // assert(iBucketIndex < NUM_BUCKET);

                    // Push points into the bucket
                    if ((int)minQue.size() < PARAM_LSH_NUM_INDEX_PROBES)
                        minQue.push(IFPair(iBucketIndex, fAbsSumHash));

                    else if (fAbsSumHash > minQue.top().m_fValue)
                    {
                        minQue.pop();
                        minQue.push(IFPair(iBucketIndex, fAbsSumHash));
                    }
                }
            }

            /**
            Insert point (n, absProjectionValue) into a bucket presented as a priority queue
            We will have to extract top-(\alpha * B) points in this bucket later.
            **/

            while (!minQue.empty())
            {
                IFPair ifPair = minQue.top(); // index is bucketID, value is sumAbsHash
                minQue.pop();
                vecBucket_MaxQue[ifPair.m_iIndex].push(IFPair(n, ifPair.m_fValue));
            }
        }

        // Only for debug: count # points in a table
//        int iNumPoint = 0;
//        for (int i = 0; i < NUM_BUCKET; ++i)
//        {
//            iNumPoint += vecBucket_MaxQue[i].size();
//        }
//
//        assert(iNumPoint == PARAM_DATA_N * PARAM_LSH_NUM_INDEX_PROBES * PARAM_LSH_NUM_INDEX_PROBES);

        // Convert priorityQueue to vector
        // Now each bucket is a vector sorted by the hash value (projected value)
        // REQUIREMENT: Largest value are at the front of the vector
        int iNumPoint = 0;
        for (int h = 0; h < NUM_BUCKET; ++h )
        {
            // NOTE: must check empty bucket
            if (vecBucket_MaxQue[h].empty())
                continue;

            int iBucketSize = vecBucket_MaxQue[h].size();
            vector<int> vecBucket(iBucketSize, -1); // hack: use pointIdx = -1 to find the bug if happen

            // Since the queue pop the max value first
            for (int i = 0; i < iBucketSize; ++i )
            {
                vecBucket[i] = vecBucket_MaxQue[h].top().m_iIndex;
                vecBucket_MaxQue[h].pop();
            }

            // We must scale to make sure that the number of points is: alpha * N
            int iLimit = (int)ceil(PARAM_LSH_BUCKET_SIZE_SCALE * iBucketSize / PARAM_LSH_NUM_INDEX_PROBES);

            // If the bucket is small, we do not scale it much, keep at least top-K candidates
            // In practice, it is useful
            // For a fair comparison with threshold LSF or Falconn, this IF need to be deleted
            if (PARAM_INTERNAL_LIMIT_BUCKET && (iLimit < PARAM_MIPS_TOP_K))
            {
                iLimit = min(PARAM_MIPS_TOP_K, iBucketSize);
                iLowerBound_Count++; // counting number of small buckets that are not fully scaled by \alpha
            }

            iNumPoint += iLimit;

            /** Build data structure here
            - must set #pragma omp critical since changing one bucket affects other buckets (2D vector would be safer for threading)

            - VECTOR_PAIR_FALCONN_BUCKET_POS: contains pair<beginning of bucket, # points in the bucket>
            Empty bucket will be stored as well with iLimit = 0
            - VECTOR_FALCONN_TABLES: an 1D array contains non-empty buckets, each bucket contain a vector of point Idx (sorted by abs(projValue))
            **/
            // First: update the position of the bucket
            // Must use uint32_t since we might deal with large L
            // e.g. L = 2^10 * (2^10 * 2^10) = 2^32
            #pragma omp critical
            {
                VECTOR_PAIR_FALCONN_BUCKET_POS[iBaseTableIdx + h] = make_pair((uint32_t)VECTOR_FALCONN_TABLES.size(), (uint16_t)iLimit);

    //            #pragma omp critical
//                int temp = VECTOR_FALCONN_TABLES.size();

                // Second: add the bucket into a 1D vector
                // This is the global data structure, it must be declared as critical
                //#pragma omp critical
                VECTOR_FALCONN_TABLES.insert(VECTOR_FALCONN_TABLES.end(), vecBucket.begin(), vecBucket.begin() + iLimit);

    //            #pragma omp critical
//                assert(VECTOR_FALCONN_TABLES.size() - temp == iLimit);
            }


        }

        fScaleData += (1.0 * iNumPoint / PARAM_DATA_N) / PARAM_LSH_NUM_TABLE;
	}

	//shink_to_fit
	VECTOR_FALCONN_TABLES.shrink_to_fit();
	VECTOR_PAIR_FALCONN_BUCKET_POS.shrink_to_fit();

//	cout << "Finish building index... " << endl;
//    cout << "Size of VECTOR_FALCONN_TABLES using sizeof() in bytes: " << sizeof(VECTOR_FALCONN_TABLES) << endl;
//    cout << "Size of an element in bytes: " << sizeof(VECTOR_FALCONN_TABLES[0]) << endl;
//    cout << "Number of element: " << VECTOR_FALCONN_TABLES.size() << endl;

    double dTemp = 1.0 * sizeof(VECTOR_FALCONN_TABLES)  / (1 << 30) +
                   1.0 * sizeof(VECTOR_FALCONN_TABLES[0]) * VECTOR_FALCONN_TABLES.size() / (1 << 30) ; // capacity() ?

//    cout << "Size of VECTOR_FALCONN_TABLES in GB by sum sizeof() + capacity() * 4: " << dTemp << endl;

    double dIndexSize = dTemp; // convert to GB


//    cout << "Size of VECTOR_PAIR_FALCONN_BUCKET_POS using sizeof() in bytes:  " << sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS) << endl;
//    cout << "Size of element in bytes: " << sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS[0]) << endl;
//    cout << "Number of lement: " << VECTOR_PAIR_FALCONN_BUCKET_POS.size() << endl;

    dTemp = 1.0 * sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS) / (1 << 30) +
            1.0 * sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS[0]) * VECTOR_PAIR_FALCONN_BUCKET_POS.size() / (1 << 30); // in GB

//    cout << "Size of VECTOR_PAIR_FALCONN_BUCKET_POS in GB by sum sizeof() + capacity() * 4: " << dTemp << endl;

    dIndexSize += dTemp;  // convert to GB

    cout << "Size of 1D ScaledFalconnCEOs2 index in GB: " << dIndexSize << endl;

    cout << "numPoints in Table / N: " << fScaleData << endl;
    cout << "Percentage of partially scaled buckets in a table: " << 1.0 * iLowerBound_Count / (NUM_BUCKET * PARAM_LSH_NUM_TABLE) << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Construct 1D ScaledFalconnCEOs2 Data Structure Wall Time (in seconds): " << (float)duration.count() << " seconds" << endl;
}

MatrixXi FalconnPP::query(MatrixXf& queries, int count, int probes) {
//    cout << "Scaled FalconnCEOs Cyclic Probes querying..." << endl;
// PARAM_LSH_NUM_QUERY_PROBES = probes;
    PARAM_LSH_NUM_QUERY_PROBES = PARAM_TEST_LSH_qPROBE_BASE + PARAM_TEST_LSH_qPROBE_RANGE * probes;
    PARAM_MIPS_TOP_K = count;
    PARAM_QUERY_Q = queries.cols();

    MATRIX_Q = queries;

    auto startTime = chrono::high_resolution_clock::now();

    // useful for debug
//    float hashTime = 0, lookupTime = 0, distTime = 0;
//	uint64_t iTotalProbes = 0, iTotalUniqueCand = 0, iTotalCand = 0;
//    int iNumEmptyBucket = 0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    int NUM_BUCKET = PARAM_INTERNAL_LSH_NUM_BUCKET * PARAM_INTERNAL_LSH_NUM_BUCKET;


    // Trick: Only sort to get top-maxProbe since we do not need the rest.
    // This will reduce the cost of LDlogD to LDlog(maxProbe) for faster querying
    // Otherwise, we have to use on-the-fly sorting (similar to Falconn)
    // 2.0* should have enough number of probes per 1 LSH function to extract the top-k projection values
    int iMaxProbesPerTable = ceil(2.0 * PARAM_LSH_NUM_QUERY_PROBES / PARAM_LSH_NUM_TABLE);
    int iMaxProbesPerHash = ceil(sqrt(1.0 * iMaxProbesPerTable)); // since we combine 2 LSH functions

//    cout << "Max probes per table is " << iMaxProbesPerTable << endl;
//    cout << "Max probes per rotation is " << iMaxProbesPerRotate << endl;

//    int num_threads = 64;
//    omp_set_num_threads(num_threads); //set the number of threads
//
//    #pragma omp parallel for reduction(+:hashTime, lookupTime, distTime, iTotalProbes, iTotalUniqueCand, iTotalCand) //num_threads(num_threads)
    #pragma omp parallel for
	for (int q = 0; q < PARAM_QUERY_Q; ++q)
	{
//		auto startTime = chrono::high_resolution_clock::now();

		// Get hash value of all hash table first
		VectorXf vecQuery = MATRIX_Q.col(q);

		// For each hash table, keep top-m largest projections for query probing
		// We use a priority queue to keep track the projection value
		vector<priority_queue< IFPair, vector<IFPair>, greater<IFPair> >> vecMinQue(PARAM_LSH_NUM_TABLE);

		/** Rotating and prepared probes sequence **/
		for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            VectorXf rotatedQ1 = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
            rotatedQ1.segment(0, PARAM_DATA_D) = vecQuery;

            VectorXf rotatedQ2 = rotatedQ1;

            for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
            {

                for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
                {
                    rotatedQ1(d) *= (2 * (int)bitHD1[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                    rotatedQ2(d) *= (2 * (int)bitHD2[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                }

                fht_float(rotatedQ1.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
                fht_float(rotatedQ2.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
            }


            // Assign hashIndex and compute distance between hashValue and the maxValue
            // Then insert into priority queue
            // Get top-k max position on each rotations
            // minQueue might be better regarding space usage, hence better for cache
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQue1;
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQue2;

            for (int r = 0; r < PARAM_LSH_NUM_PROJECTION; ++r)
            {
                // 1st LSH
                int iSign = sgn(rotatedQ1(r));
                float fHashValue = iSign * rotatedQ1(r);

                // Get hashIndex
                int iHashIndex = r;
                if (iSign < 0)
                    iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION;

                // cout << "Hash index 1 : " << iBucketIndex << endl;

                // Trick: We only keep top-iMaxProbesPerHash since
                // Falconn uses block sorting to save sorting time
                if ((int)minQue1.size() < iMaxProbesPerHash)
                    minQue1.push(IFPair(iHashIndex, fHashValue));

                // queue is full
                else if (fHashValue > minQue1.top().m_fValue)
                {
                    minQue1.pop(); // pop max, and push min hash distance
                    minQue1.push(IFPair(iHashIndex, fHashValue));
                }

                // 2nd LSH
                iSign = sgn(rotatedQ2(r));
                fHashValue = iSign * rotatedQ2(r);

                // Get hashIndex
                iHashIndex = r;
                if (iSign < 0)
                    iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION;

                // cout << "Hash index 2: " << iBucketIndex << endl;

                // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                // Falconn uses block sorting to save sorting time
                if ((int)minQue2.size() < iMaxProbesPerHash)
                    minQue2.push(IFPair(iHashIndex, fHashValue));

                // queue is full
                else if (fHashValue > minQue2.top().m_fValue)
                {
                    minQue2.pop(); // pop max, and push min hash distance
                    minQue2.push(IFPair(iHashIndex, fHashValue));
                }
            }

//            assert((int)minQue1.size() == iMaxProbesPerRotate);
//            assert((int)minQue2.size() == iMaxProbesPerRotate);

            // Convert to vector, the large projection value is in [0]
            // Hence better for creating a sequence of probing since we do not have to call pop() many times
            vector<IFPair> vec1(iMaxProbesPerHash), vec2(iMaxProbesPerHash);
            for (int p = iMaxProbesPerHash - 1; p >= 0; --p)
            {
                // 1st LSH
                IFPair ifPair = minQue1.top();
                minQue1.pop();
                vec1[p] = ifPair;

                // 2nd LSH
                ifPair = minQue2.top();
                minQue2.pop();
                vec2[p] = ifPair;
            }

            // Now begin building the query probes on ONE table
            for (const auto& ifPair1: vec1)
            {
                int iHashIndex1 = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                //cout << "Hash index 1: " << iBucketIndex1 << " projection value: " << fAbsHashValue1 << endl;

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iHashIndex2 = ifPair2.m_iIndex;
                    float fAbsHashValue2 = ifPair2.m_fValue;

                    //cout << "Hash index 2: " << iBucketIndex2 << " projection value: " << fAbsHashValue2 << endl;

                    // Start building the probe sequence
                    int iBucketIndex = iHashIndex1 * PARAM_INTERNAL_LSH_NUM_BUCKET + iHashIndex2; // (totally we have 2D * 2D buckets)
                    float fSumHashValue = fAbsHashValue1 + fAbsHashValue2;

                    assert(iBucketIndex < NUM_BUCKET);

                    // IMPORTANT: Must use ALL iMaxProbesPerTable < iMaxProbesPerRotate^2
                    // since the minQueue will pop the min projection value first
                    // If do not use iMaxProbesPerRotate^2, we miss the bucket of query (max + max)
                    if ((int)vecMinQue[l].size() < iMaxProbesPerTable)
                        vecMinQue[l].push(IFPair(iBucketIndex, fSumHashValue));

                    else if (fSumHashValue > vecMinQue[l].top().m_fValue)
                    {
                        vecMinQue[l].pop(); // pop max, and push min hash distance
                        vecMinQue[l].push(IFPair(iBucketIndex, fSumHashValue));
                    }
                }
            }

//            assert((int)vecMinQue[l].size() == iMaxProbesPerTable);

        }


        // We need to dequeue to get the bucket on the right order
        // Every table has iMaxProbes positions for query probing
        // TODO: use Boost.MultiArray for less maintaining cost, and perhap cache-friendly
        vector<IFPair> vecBucketProbes(PARAM_LSH_NUM_TABLE * iMaxProbesPerTable);
        for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            int iBaseTableIdx = l * NUM_BUCKET;
            int iBaseIdx = l * iMaxProbesPerTable; // this is for vecProbes

            int idx = iMaxProbesPerTable - 1;

            while (!vecMinQue[l].empty())
            {
                // m_iIndex = bucketIndex, mfValue = absHashValue1 + absHashValue2
                IFPair ifPair = vecMinQue[l].top();
                vecMinQue[l].pop();

                //cout << "Bucket index: " << ifPair.m_iIndex << endl;

                // Now: ifPair.m_iIndex is the bucket index
                // changing the index to have TableIdx information since we iterate probing through all tables
                // This index is used to access the hash table VECTOR_FALCONN_TABLES
                ifPair.m_iIndex = iBaseTableIdx + ifPair.m_iIndex;
                // Now: ifPair.m_iIndex contains the position of the table idx & hashIndex

                // Since we pop from minQueue, it need to be stored at the end first (i.e. idx = iMaxProbesPerTable - 1)
                vecBucketProbes[iBaseIdx + idx] = ifPair;
                idx--;
            }

//            printVector(vecProbes[l]);

        }


		// Then preparing multi-probe
        // We get the bucket of query in all tables first
        priority_queue< IFPair, vector<IFPair> > maxQueProbes;
        for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            // position of query's buckets
            // Query probe criteria: abs(<q, ri>) + abs(<q, si>)
            // The closer q and (ri, si), the better chance to get kNN
            maxQueProbes.push(vecBucketProbes[l * iMaxProbesPerTable]);
        }

//        auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
//        hashTime += (float)durTime.count();

        /** Query probing **/
//        startTime = chrono::high_resolution_clock::now();

		boost::dynamic_bitset<> bitsetHist(PARAM_DATA_N); // all 0's by default
        VectorXi vecProbeTracking = VectorXi::Zero(PARAM_LSH_NUM_TABLE);

        int iNumCand = 0;
        for (int iProbeCount = 0; iProbeCount < PARAM_LSH_NUM_QUERY_PROBES; iProbeCount++)
        {

//            iTotalProbes++;

            // maxQueProbes has all query's buckets of L tables, and they will be used first
            // After using a query's bucket of table T, we insert into maxQueProbes the next bucket of T.
            // The maxQueProbes will determine the next bucket of the hash table to use.
            IFPair ifPair = maxQueProbes.top();
            maxQueProbes.pop();
//            cout << "Probe " << iProbeCount << ": " << ifPair.m_iIndex << " " << ifPair.m_fValue << endl;

            // ifPair.m_iIndex is the (tableIdx & bucketIdx)
            uint32_t iBucketPos = VECTOR_PAIR_FALCONN_BUCKET_POS[ifPair.m_iIndex].first;
            uint16_t iBucketSize = VECTOR_PAIR_FALCONN_BUCKET_POS[ifPair.m_iIndex].second;

            // Update probe tracking
            int iTableIdx = ifPair.m_iIndex / NUM_BUCKET; // get table idx

            //cout << "Table: " << iTableIdx << " Hash index: " << ifPair.m_iIndex - iTableIdx * NUM_BUCKET << endl;
            //printVector(vecBucket);

            vecProbeTracking(iTableIdx)++;

            // insert into the queue for next probes
            if (vecProbeTracking(iTableIdx) < iMaxProbesPerTable)
            {
                // vecBucketProbes has range l * iMaxProbesPerTable + idx (ie top-probes)
                IFPair ifPair = vecBucketProbes[iTableIdx * iMaxProbesPerTable + vecProbeTracking(iTableIdx)]; // get the next bucket idx of the investigated hash table

                // Query probing criteria: abs(<q, ri>) + abs(<q, si>)
                // The closer q and (ri, si), the better chance to get NN
                // It is identical to the CEOs property to get top-s0 min & max vectors
                maxQueProbes.push(ifPair);
            }

            if (iBucketSize == 0)
            {
//                iNumEmptyBucket++;
                continue;
            }

    /**
	For debugging
	**/
//            iTotalCand += iBucketSize;

            // Get all points in the bucket
            for (int i = 0; i < iBucketSize; ++i)
            {
                int iPointIdx = VECTOR_FALCONN_TABLES[iBucketPos + i];

//                assert (iPointIdx < PARAM_DATA_N);
//                assert (iPointIdx >= 0);

                if (~bitsetHist[iPointIdx])
                {
                    iNumCand++;
                    bitsetHist[iPointIdx] = 1;
                }

            }

            // Allow all query probes before reaching the limit
//            if (iNumCand >= PARAM_MIPS_CANDIDATE_SIZE)
//                break;
		}

    /**
	For debugging
	**/
//        durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
//        lookupTime += (float)durTime.count();

//        iTotalProbes += iProbeCount;
//		iTotalUniqueCand += iNumCand; // bitsetHist.count();

//		startTime = chrono::high_resolution_clock::now();

		// in case some query does not have candidate or qProbes is too small
        if (iNumCand == 0)
            continue;

		// This is to get top-K
		priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

        //new A class to do inner product by SIMD from HnswLib
//        hnswlib::InnerProductSpace inner(PARAM_DATA_D);
//        hnswlib::DISTFUNC<float> fstdistfunc_ = inner.get_dist_func();
//        void *dist_func_param_;
//        dist_func_param_ = inner.get_dist_func_param();


        size_t iPointIdx = bitsetHist.find_first();
        while (iPointIdx != boost::dynamic_bitset<>::npos)
        {
            // Get dot product
//            float fInnerProduct = fstdistfunc_(vecQuery.data(), MATRIX_X.col(iPointIdx).data(), dist_func_param_);
            float fInnerProduct = vecQuery.dot(MATRIX_X.col(iPointIdx));

            // Add into priority queue
            if (int(minQueTopK.size()) < PARAM_MIPS_TOP_K)
                minQueTopK.push(IFPair(iPointIdx, fInnerProduct));

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fInnerProduct));
            }

            iPointIdx = bitsetHist.find_next(iPointIdx);
        }

//        cout << "Queue TopK size: " << minQueTopK.size() << endl;

        // assert((int)minQueTopK.size() == PARAM_MIPS_TOP_K);

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
//            cout << "Bug is here at " << k << endl;

            matTopK(k, q) = minQueTopK.top().m_iIndex;
            minQueTopK.pop();
        }

    /**
	For debugging
	**/
//		durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
//		distTime += (float)durTime.count();
	}

	auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

	/**
	For debugging
	**/
//	cout << "Finish querying..." << endl;
//    cout << "Average number of empty buckets per query: " << (double)iNumEmptyBucket / PARAM_QUERY_Q << endl;
//	cout << "Average number of probes per query: " << (double)iTotalProbes / PARAM_QUERY_Q << endl;
//	cout << "Average number of unique candidates per query: " << (double)iTotalUniqueCand / PARAM_QUERY_Q << endl;
//	cout << "Average number of candidates per query: " << (double)iTotalCand / PARAM_QUERY_Q << endl;

//	cout << "Hash and Probing Time: " << hashTime << " ms" << endl;
//	cout << "Lookup Time: " << lookupTime << " ms" << endl;
//	cout << "Distance Time: " << distTime << " ms" << endl;

	// cout << "Querying Time: " << (float)durTime.count() << " ms" << endl;


	if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName = PARAM_OUTPUT_FILE + "_TopK_" + int2str(PARAM_MIPS_TOP_K) +
                        "_NumProjection_" + int2str(PARAM_LSH_NUM_PROJECTION) +
                        "_NumTable_" + int2str(PARAM_LSH_NUM_TABLE) +
                        "_IndexProbe_"  + int2str(PARAM_LSH_NUM_INDEX_PROBES) +
                        "_QueryProbe_"  + int2str(PARAM_LSH_NUM_QUERY_PROBES) +
                        "_BucketScale_"  + int2str((int)(PARAM_LSH_BUCKET_SIZE_SCALE * 100)) +
                        "_CandidateSize_" + int2str(PARAM_MIPS_CANDIDATE_SIZE) + ".txt";

        outputFile(matTopK, sFileName);
	}
    return matTopK;

}
