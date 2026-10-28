//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//              Apple Metal backend                       //
//********************************************************//

// Batches of ExtractSift, MatchSiftData, FindHomographyGPU and ImproveHomographyGPU
// calls (Metal backend).
//
// Each CudaSift call is submitted to the GPU and waited for separately, which costs
// 130-160 us per call on an Apple M5 Pro, mostly driver latency. Calls made between
// BeginSiftBatch() and EndSiftBatch() are recorded and submitted together instead,
// e.g. the extraction of a new frame and its matching with the previous frame:
//
//   BeginSiftBatch();
//   ExtractSift(siftData2, img2, 5, 1.0f, 3.0f);
//   MatchSiftData(siftData1, siftData2);
//   EndSiftBatch();      // submits, returns immediately
//   ...                  // other CPU work
//   WaitSiftBatch();     // results are available
//
// The calls run in order, with the same results as outside a batch. Until the batch
// completes (WaitSiftBatch() or a call outside a batch, see below), the results
// (SiftData::numPts, the points, the match fields, the homographies and their counts)
// are not valid, and the images, SiftData and homographies used must not be modified
// or freed.
//
// - FindHomography and FindHomographySeeded cannot be called inside a batch (they sample
//   the matches on the CPU): the program stops with an error. FindHomographyGPU can.
// - Inside a batch, MatchSiftData, FindHomographyGPU and ImproveHomographyGPU
//   return 0.0 (the time is not available).
// - BeginSiftBatch() inside an open batch stops the program with an error.
//   WaitSiftBatch() ends an open batch (EndSiftBatch) before waiting for it.
// - A call made outside a batch (ExtractSift, MatchSiftData, FindHomography,
//   FindHomographyGPU, ImproveHomographyGPU) after EndSiftBatch() first waits for
//   the batch and completes it, so its results become valid. WaitSiftBatch() still
//   returns the batch's GPU time. CPU-only functions (ImproveHomography, PrintSiftData)
//   don't wait: call WaitSiftBatch() first.

#ifndef CUDASIFTBATCH_H
#define CUDASIFTBATCH_H

void BeginSiftBatch();
void EndSiftBatch();
double WaitSiftBatch();   // returns the GPU time of the batch in ms

#endif
