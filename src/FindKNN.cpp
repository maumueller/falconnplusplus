#include <chrono>
#include <iostream>
#include <queue>
#include <fstream>
#include <string>

#include "FindKNN.h"
#include "Header.h"
#include "Utilities.h"

/**
The adaptive probing will utilize more hash table and hence returns higher accuracy given the same candidate size
**/
void simpleFalconnCEOsTopK_CycProbes2_1D()
{
//    cout << "Scaled FalconnCEOs Cyclic Probes querying..." << endl;

    auto startTime = chrono::high_resolution_clock::now();

    float hashTime = 0, lookupTime = 0, distTime = 0;
	uint64_t iTotalProbes = 0, iTotalUniqueCand = 0, iTotalCand = 0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    int NUM_BUCKET = PARAM_INTERNAL_LSH_NUM_BUCKET * PARAM_INTERNAL_LSH_NUM_BUCKET;
    int iNumEmptyBucket = 0;

    // Trick: Only sort to get top-maxProbe since we do not need the rest.
    // This will reduce the cost of LDlogD to LDlog(maxProbe) for faster querying
    // 4.0* should have enough number of probes per rotation to extract the top-k projection values
    int iMaxProbesPerTable = ceil(4.0 * PARAM_LSH_NUM_QUERY_PROBES / PARAM_LSH_NUM_TABLE);
    int iMaxProbesPerRotate = ceil(sqrt(1.0 * iMaxProbesPerTable));

//    cout << "Max probes per table is " << iMaxProbesPerTable << endl;
//    cout << "Max probes per rotation is " << iMaxProbesPerRotate << endl;

    #pragma omp parallel for reduction(+:hashTime, lookupTime, distTime, iTotalProbes, iTotalUniqueCand, iTotalCand)
	for (int q = 0; q < PARAM_QUERY_Q; ++q)
	{
		auto startTime = chrono::high_resolution_clock::now();

		// Get hash value of all hash table first
		VectorXf vecQuery = MATRIX_Q.col(q);

		// Contain top-m largest projections for each hash table
		// We use a priority queue to keep track the projection value
		vector<priority_queue< IFPair, vector<IFPair>, greater<IFPair> >> vecMinQue(PARAM_LSH_NUM_TABLE);

		/** Rotating and prepared probes sequence **/
		for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            VectorXf rotatedQ1 = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
            rotatedQ1.segment(0, PARAM_DATA_D) = vecQuery;

            VectorXf rotatedQ2 = rotatedQ1;

//            MatrixXf matHD1 = MATRIX_HD1.middleCols(l * PARAM_NUM_ROTATION, PARAM_NUM_ROTATION);
//            MatrixXf matHD2 = MATRIX_HD2.middleCols(l * PARAM_NUM_ROTATION, PARAM_NUM_ROTATION);

            for (int r = 0; r < PARAM_NUM_ROTATION; ++r)
            {

                for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
                {
                    rotatedQ1(d) *= (2 * (int)bitHD1[l * PARAM_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                    rotatedQ2(d) *= (2 * (int)bitHD2[l * PARAM_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                }

//                rotatedQ1 = rotatedQ1.cwiseProduct(matHD1.col(r));
//                rotatedQ2 = rotatedQ2.cwiseProduct(matHD2.col(r));

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
                // 1st rotation
                int iSign = sgn(rotatedQ1(r));
                float fHashDiff = iSign * rotatedQ1(r);

                // Get hashIndex
                int iBucketIndex = r;
                if (iSign < 0)
                    iBucketIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION;

                // cout << "Hash index 1 : " << iBucketIndex << endl;

                // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                // Falconn uses block sorting to save sorting time
                if ((int)minQue1.size() < iMaxProbesPerRotate)
                    minQue1.push(IFPair(iBucketIndex, fHashDiff));

                // queue is full
                else if (fHashDiff > minQue1.top().m_fValue)
                {
                    minQue1.pop(); // pop max, and push min hash distance
                    minQue1.push(IFPair(iBucketIndex, fHashDiff));
                }

                // 2nd rotation
                iSign = sgn(rotatedQ2(r));
                fHashDiff = iSign * rotatedQ2(r);

                // Get hashIndex
                iBucketIndex = r;
                if (iSign < 0)
                    iBucketIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION;

                // cout << "Hash index 2: " << iBucketIndex << endl;

                // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                // Falconn uses block sorting to save sorting time
                if ((int)minQue2.size() < iMaxProbesPerRotate)
                    minQue2.push(IFPair(iBucketIndex, fHashDiff));

                // queue is full
                else if (fHashDiff > minQue2.top().m_fValue)
                {
                    minQue2.pop(); // pop max, and push min hash distance
                    minQue2.push(IFPair(iBucketIndex, fHashDiff));
                }
            }

//            assert((int)minQue1.size() == iMaxProbesPerRotate);
//            assert((int)minQue2.size() == iMaxProbesPerRotate);

            // Convert to vector, the large projection value is in [0]
            // Hence better for creating a sequence of probing since we do not have to call pop() many times
            vector<IFPair> vec1(iMaxProbesPerRotate), vec2(iMaxProbesPerRotate);
            for (int p = iMaxProbesPerRotate - 1; p >= 0; --p)
            {
                // 1st rotation
                IFPair ifPair = minQue1.top();
                minQue1.pop();
                vec1[p] = ifPair;

                // 2nd rotation
                ifPair = minQue2.top();
                minQue2.pop();
                vec2[p] = ifPair;
            }

            // Now begin building the query probes on ONE table
            for (const auto& ifPair1: vec1)
            {
                int iBucketIndex1 = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                //cout << "Hash index 1: " << iBucketIndex1 << " projection value: " << fAbsHashValue1 << endl;

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iBucketIndex2 = ifPair2.m_iIndex;
                    float fAbsHashValue2 = ifPair2.m_fValue;

                    //cout << "Hash index 2: " << iBucketIndex2 << " projection value: " << fAbsHashValue2 << endl;

                    // Start building the probe sequence
                    int iBucketIndex = iBucketIndex1 * PARAM_INTERNAL_LSH_NUM_BUCKET + iBucketIndex2; // (totally we have 2D * 2D buckets)
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
        // TODO: use Boost.MultiArray for less maintaining cost
        vector<IFPair> vecBucketProbes(PARAM_LSH_NUM_TABLE * iMaxProbesPerTable);
        for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            int iBaseTableIdx = l * NUM_BUCKET;
            int iBaseIdx = l * iMaxProbesPerTable; // this is for vecProbes

            int idx = iMaxProbesPerTable - 1;

            while (!vecMinQue[l].empty())
            {
                // m_iIndex = hashIndex, mfValue = absHashValue1 + absHashValue2
                IFPair ifPair = vecMinQue[l].top();
                vecMinQue[l].pop();

                //cout << "Hash index: " << ifPair.m_iIndex << endl;

                // Now: ifPair.m_iIndex is the hash Index (ie random vector with sign)
                // changing the index to have TableIdx information since we iterate probing through all tables
                // This index is used to access the hash table VECTOR_FALCONN_TABLES
                ifPair.m_iIndex = iBaseTableIdx + ifPair.m_iIndex;
                // Now: ifPair.m_iIndex contains the position of the table idx & hashIndex

                vecBucketProbes[iBaseIdx + idx] = ifPair;
                idx--;
            }

//            printVector(vecProbes[l]);

        }


		// Then preparing multi-probe
        // for all probes, we select the min distance or -abs(projectedValue) first
        priority_queue< IFPair, vector<IFPair> > maxQueProbes;
        for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            maxQueProbes.push(vecBucketProbes[l * iMaxProbesPerTable]); // position of query of all tables
        }

        auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
        hashTime += (float)durTime.count();

        /** Querying **/
        startTime = chrono::high_resolution_clock::now();

		boost::dynamic_bitset<> bitsetHist(PARAM_DATA_N); // all 0's by default
        VectorXi vecProbeTracking = VectorXi::Zero(PARAM_LSH_NUM_TABLE);

        int iNumCand = 0;
        for (int iProbeCount = 0; iProbeCount < PARAM_LSH_NUM_QUERY_PROBES; iProbeCount++)
        {

            iTotalProbes++;

            IFPair ifPair = maxQueProbes.top();
            maxQueProbes.pop();
//            cout << "Probe " << iProbeCount << ": " << ifPair.m_iIndex << " " << ifPair.m_fValue << endl;

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
                maxQueProbes.push(ifPair);
            }

            if (iBucketSize == 0)
            {
                iNumEmptyBucket++;
                continue;
            }

            iTotalCand += iBucketSize;

            // Get all points in the bucket
            for (int i = 0; i < iBucketSize; ++i)
            {
                int iPointIdx = VECTOR_FALCONN_TABLES[iBucketPos + i];

                assert (iPointIdx < PARAM_DATA_N);
                assert (iPointIdx >= 0);

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

        durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
        lookupTime += (float)durTime.count();

//        iTotalProbes += iProbeCount;
		iTotalUniqueCand += iNumCand; // bitsetHist.count();

		startTime = chrono::high_resolution_clock::now();

//		matTopK.col(q) = computeSimilarity(setCandidates, q);
//		cout << "Nuber of bit set: " << setHistogram.count() << endl;

		// getTopK(bitsetHist, vecQuery, matTopK.col(q));

//        cout << "Number of candidate: " << bitsetHist.count() << endl;
        if (iNumCand == 0)
            continue;

//        cout << "Bug here...: " << bitsetHist.count() << endl;

		// This is to get top-K
		priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

		//#pragma omp parallel for

        size_t iPointIdx = bitsetHist.find_first();
        while (iPointIdx != boost::dynamic_bitset<>::npos)
        {
            // Get dot product
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

		durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
		distTime += (float)durTime.count();
	}

	auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

//	cout << "Finish querying..." << endl;
    cout << "Average number of empty buckets per query: " << (double)iNumEmptyBucket / PARAM_QUERY_Q << endl;
	cout << "Average number of probes per query: " << (double)iTotalProbes / PARAM_QUERY_Q << endl;
	cout << "Average number of unique candidates per query: " << (double)iTotalUniqueCand / PARAM_QUERY_Q << endl;
	cout << "Average number of candidates per query: " << (double)iTotalCand / PARAM_QUERY_Q << endl;

	cout << "Hash and Probing Time: " << hashTime << " ms" << endl;
	cout << "Lookup Time: " << lookupTime << " ms" << endl;
	cout << "Distance Time: " << distTime << " ms" << endl;
	cout << "Querying Time: " << (float)durTime.count() << " ms" << endl;


	if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName;

        if (PARAM_INTERNAL_LIMIT_BUCKET)

            sFileName = "1DFalconnCEOsCyc2_Top_" + int2str(PARAM_MIPS_TOP_K) +
                        "_NumProjection_" + int2str(PARAM_LSH_NUM_PROJECTION) +
                        "_NumTable_" + int2str(PARAM_LSH_NUM_TABLE) +
                        "_IndexProbe_"  + int2str(PARAM_LSH_NUM_INDEX_PROBES) +
                        "_QueryProbe_"  + int2str(PARAM_LSH_NUM_QUERY_PROBES) +
                        "_BucketScale_"  + int2str((int)(PARAM_LSH_BUCKET_SIZE_SCALE * 100)) +
                        "_CandidateSize_" + int2str(PARAM_MIPS_CANDIDATE_SIZE) + ".txt";

        else

            sFileName = "1DFalconnCEOsCyc2_Top_NoLimit_" + int2str(PARAM_MIPS_TOP_K) +
                        "_NumProjection_" + int2str(PARAM_LSH_NUM_PROJECTION) +
                        "_NumTable_" + int2str(PARAM_LSH_NUM_TABLE) +
                        "_IndexProbe_"  + int2str(PARAM_LSH_NUM_INDEX_PROBES) +
                        "_QueryProbe_"  + int2str(PARAM_LSH_NUM_QUERY_PROBES) +
                        "_BucketScale_"  + int2str((int)(PARAM_LSH_BUCKET_SIZE_SCALE * 100)) +
                        "_CandidateSize_" + int2str(PARAM_MIPS_CANDIDATE_SIZE) + ".txt";


        outputFile(matTopK, sFileName);
	}

}


/**
The adaptive probing will utilize more hash table and hence returns higher accuracy given the same candidate size
**/
void thresFalconnCEOsTopK_CycProbes2_1D()
{
//    cout << "Scaled FalconnCEOs Cyclic Probes querying..." << endl;

    auto startTime = chrono::high_resolution_clock::now();

    float hashTime = 0, lookupTime = 0, distTime = 0;
	uint64_t iTotalProbes = 0, iTotalUniqueCand = 0, iTotalCand = 0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    int NUM_BUCKET = PARAM_INTERNAL_LSH_NUM_BUCKET * PARAM_INTERNAL_LSH_NUM_BUCKET;
    int iNumEmptyBucket = 0;

    // Trick: Only sort to get top-maxProbe since we do not need the rest.
    // This will reduce the cost of LDlogD to LDlog(maxProbe) for faster querying
    // 4.0* should have enough number of probes per rotation to extract the top-k projection values
    int iMaxProbesPerTable = ceil(4.0 * PARAM_LSH_NUM_QUERY_PROBES / PARAM_LSH_NUM_TABLE);
    int iMaxProbesPerRotate = ceil(sqrt(1.0 * iMaxProbesPerTable));

//    cout << "Max probes per table is " << iMaxProbesPerTable << endl;
//    cout << "Max probes per rotation is " << iMaxProbesPerRotate << endl;

    #pragma omp parallel for reduction(+:hashTime, lookupTime, distTime, iTotalProbes, iTotalUniqueCand, iTotalCand)
	for (int q = 0; q < PARAM_QUERY_Q; ++q)
	{
		auto startTime = chrono::high_resolution_clock::now();

		// Get hash value of all hash table first
		VectorXf vecQuery = MATRIX_Q.col(q);

		// Contain top-m largest projections for each hash table
		// We use a priority queue to keep track the projection value
		vector<priority_queue< IFPair, vector<IFPair>, greater<IFPair> >> vecMinQue(PARAM_LSH_NUM_TABLE);

		/** Rotating and prepared probes sequence **/
		for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            MatrixXf matHD1 = MATRIX_HD1.middleCols(l * PARAM_NUM_ROTATION, PARAM_NUM_ROTATION);
            MatrixXf matHD2 = MATRIX_HD2.middleCols(l * PARAM_NUM_ROTATION, PARAM_NUM_ROTATION);

            VectorXf rotatedQ1 = VectorXf::Zero(PARAM_LSH_NUM_PROJECTION);
            rotatedQ1.segment(0, PARAM_DATA_D) = vecQuery;

            VectorXf rotatedQ2 = rotatedQ1;

            for (int r = 0; r < PARAM_NUM_ROTATION; ++r)
            {
                rotatedQ1 = rotatedQ1.cwiseProduct(matHD1.col(r));
                fht_float(rotatedQ1.data(), PARAM_INTERNAL_LOG2_NUM_PROJECTION);

                rotatedQ2 = rotatedQ2.cwiseProduct(matHD2.col(r));
                fht_float(rotatedQ2.data(), PARAM_INTERNAL_LOG2_NUM_PROJECTION);
            }


            // Assign hashIndex and compute distance between hashValue and the maxValue
            // Then insert into priority queue
            // Get top-k max position on each rotations
            // minQueue might be better regarding space usage, hence better for cache
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQue1;
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQue2;

            for (int r = 0; r < PARAM_LSH_NUM_PROJECTION; ++r)
            {
                // 1st rotation
                int iSign = sgn(rotatedQ1(r));
                float fHashDiff = iSign * rotatedQ1(r);

                // Get hashIndex
                int iBucketIndex = r;
                if (iSign < 0)
                    iBucketIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION;

                // cout << "Hash index 1 : " << iBucketIndex << endl;

                // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                // Falconn uses block sorting to save sorting time
                if ((int)minQue1.size() < iMaxProbesPerRotate)
                    minQue1.push(IFPair(iBucketIndex, fHashDiff));

                // queue is full
                else if (fHashDiff > minQue1.top().m_fValue)
                {
                    minQue1.pop(); // pop max, and push min hash distance
                    minQue1.push(IFPair(iBucketIndex, fHashDiff));
                }

                // 2nd rotation
                iSign = sgn(rotatedQ2(r));
                fHashDiff = iSign * rotatedQ2(r);

                // Get hashIndex
                iBucketIndex = r;
                if (iSign < 0)
                    iBucketIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION;

                // cout << "Hash index 2: " << iBucketIndex << endl;

                // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                // Falconn uses block sorting to save sorting time
                if ((int)minQue2.size() < iMaxProbesPerRotate)
                    minQue2.push(IFPair(iBucketIndex, fHashDiff));

                // queue is full
                else if (fHashDiff > minQue2.top().m_fValue)
                {
                    minQue2.pop(); // pop max, and push min hash distance
                    minQue2.push(IFPair(iBucketIndex, fHashDiff));
                }
            }

            assert((int)minQue1.size() == iMaxProbesPerRotate);
            assert((int)minQue2.size() == iMaxProbesPerRotate);

            // Convert to vector, the large projection value is in [0]
            // Hence better for creating a sequence of probing since we do not have to call pop() many times
            vector<IFPair> vec1(iMaxProbesPerRotate), vec2(iMaxProbesPerRotate);
            for (int p = iMaxProbesPerRotate - 1; p >= 0; --p)
            {
                // 1st rotation
                IFPair ifPair = minQue1.top();
                minQue1.pop();
                vec1[p] = ifPair;

                // 2nd rotation
                ifPair = minQue2.top();
                minQue2.pop();
                vec2[p] = ifPair;
            }

            // Now begin building the query probes on ONE table
            for (const auto& ifPair1: vec1)
            {
                int iBucketIndex1 = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                //cout << "Hash index 1: " << iBucketIndex1 << " projection value: " << fAbsHashValue1 << endl;

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iBucketIndex2 = ifPair2.m_iIndex;
                    float fAbsHashValue2 = ifPair2.m_fValue;

                    //cout << "Hash index 2: " << iBucketIndex2 << " projection value: " << fAbsHashValue2 << endl;

                    // Start building the probe sequence
                    int iBucketIndex = iBucketIndex1 * PARAM_INTERNAL_LSH_NUM_BUCKET + iBucketIndex2; // (totally we have 2D * 2D buckets)
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

            assert((int)vecMinQue[l].size() == iMaxProbesPerTable);

        }


        // We need to dequeue to get the bucket on the right order
        // Every table has iMaxProbes positions for query probing
        // TODO: use Boost.MultiArray for less maintaining cost
        vector<IFPair> vecBucketProbes(PARAM_LSH_NUM_TABLE * iMaxProbesPerTable);
        for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            int iBaseTableIdx = l * NUM_BUCKET;
            int iBaseIdx = l * iMaxProbesPerTable; // this is for vecProbes

            int idx = iMaxProbesPerTable - 1;

            while (!vecMinQue[l].empty())
            {
                // m_iIndex = hashIndex, mfValue = absHashValue1 + absHashValue2
                IFPair ifPair = vecMinQue[l].top();
                vecMinQue[l].pop();

                //cout << "Hash index: " << ifPair.m_iIndex << endl;

                // Now: ifPair.m_iIndex is the hash Index (ie random vector with sign)
                // changing the index to have TableIdx information since we iterate probing through all tables
                // This index is used to access the hash table VECTOR_FALCONN_TABLES
                ifPair.m_iIndex = iBaseTableIdx + ifPair.m_iIndex;
                // Now: ifPair.m_iIndex contains the position of the table idx & hashIndex

                vecBucketProbes[iBaseIdx + idx] = ifPair;
                idx--;
            }

//            printVector(vecProbes[l]);

        }


		// Then preparing multi-probe
        // for all probes, we select the min distance or -abs(projectedValue) first
        priority_queue< IFPair, vector<IFPair> > maxQueProbes;
        for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            maxQueProbes.push(vecBucketProbes[l * iMaxProbesPerTable]); // position of query of all tables
        }

        auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
        hashTime += (float)durTime.count();

        /** Querying **/
        startTime = chrono::high_resolution_clock::now();

        int iProbeCount = 0;
		boost::dynamic_bitset<> bitsetHist(PARAM_DATA_N); // all 0's by default
        VectorXi vecProbeTracking = VectorXi::Zero(PARAM_LSH_NUM_TABLE);

        while (iProbeCount < PARAM_LSH_NUM_QUERY_PROBES)
        {
            iProbeCount++;

            IFPair ifPair = maxQueProbes.top();
            maxQueProbes.pop();
//            cout << "Probe " << iProbeCount << ": " << ifPair.m_iIndex << " " << ifPair.m_fValue << endl;

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
                maxQueProbes.push(ifPair);
            }

            if (iBucketSize == 0)
            {
                iNumEmptyBucket++;
                continue;
            }

            iTotalCand += iBucketSize;

            // Get all points in the bucket
            for (int i = 0; i < iBucketSize; ++i)
            {
                int iPointIdx = VECTOR_FALCONN_TABLES[iBucketPos + i];

                assert (iPointIdx < PARAM_DATA_N);
                assert (iPointIdx >= 0);

                if (~bitsetHist[iPointIdx])
                    bitsetHist[iPointIdx] = 1;
            }

//            // Allow all query probes before reaching the limit
//            if ((int)setHistogram.count() >= PARAM_MIPS_CANDIDATE_SIZE)
//                break;
		}

        durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
        lookupTime += (float)durTime.count();

        iTotalProbes += iProbeCount;
		iTotalUniqueCand += bitsetHist.count();

		startTime = chrono::high_resolution_clock::now();

//		matTopK.col(q) = computeSimilarity(setCandidates, q);
//		cout << "Nuber of bit set: " << setHistogram.count() << endl;

		// getTopK(bitsetHist, vecQuery, matTopK.col(q));

//        cout << "Number of candidate: " << bitsetHist.count() << endl;
        if (bitsetHist.count() == 0)
            continue;

//        cout << "Bug here...: " << bitsetHist.count() << endl;

		// This is to get top-K
		priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

		//#pragma omp parallel for

        size_t iPointIdx = bitsetHist.find_first();
        while (iPointIdx != boost::dynamic_bitset<>::npos)
        {
            // Get dot product
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



		durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
		distTime += (float)durTime.count();
	}

	auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

//	cout << "Finish querying..." << endl;
    cout << "Average number of empty buckets per query: " << (double)iNumEmptyBucket / PARAM_QUERY_Q << endl;
	cout << "Average number of probes per query: " << (double)iTotalProbes / PARAM_QUERY_Q << endl;
	cout << "Average number of unique candidates per query: " << (double)iTotalUniqueCand / PARAM_QUERY_Q << endl;
	cout << "Average number of candidates per query: " << (double)iTotalCand / PARAM_QUERY_Q << endl;

	cout << "Hash and Probing Time: " << hashTime << " ms" << endl;
	cout << "Lookup Time: " << lookupTime << " ms" << endl;
	cout << "Distance Time: " << distTime << " ms" << endl;
	cout << "Querying Time: " << (float)durTime.count() << " ms" << endl;


	if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
	    string sFileName = "1DFalconnCEOsCyc2_Thres_Top_" + int2str(PARAM_MIPS_TOP_K) +
                        "_NumProjection_" + int2str(PARAM_LSH_NUM_PROJECTION) +
                        "_NumTable_" + int2str(PARAM_LSH_NUM_TABLE) +
                        "_IndexProbe_"  + int2str(PARAM_LSH_NUM_INDEX_PROBES) +
                        "_QueryProbe_"  + int2str(PARAM_LSH_NUM_QUERY_PROBES) +
                        "_BucketScale_"  + int2str((int)(PARAM_LSH_BUCKET_SIZE_SCALE * 100)) +
                        "_CandidateSize_" + int2str(PARAM_MIPS_CANDIDATE_SIZE) + ".txt";


        outputFile(matTopK, sFileName);
	}

}
