//********************************************************//
// Test: batches of calls (cudaSiftBatch.h, Metal only)   //
//********************************************************//

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftBatch.h"

int testsPassed = 0;
int testsFailed = 0;

void CHECK(bool condition, const char *msg)
{
  if (condition) {
    std::cout << "  [PASS] " << msg << std::endl;
    testsPassed++;
  } else {
    std::cout << "  [FAIL] " << msg << std::endl;
    testsFailed++;
  }
}

// Points sorted by position, scale and orientation (the point order is not deterministic)
static std::vector<SiftPoint> Sorted(SiftData &data)
{
  std::vector<SiftPoint> pts(data.h_data, data.h_data + data.numPts);
  std::sort(pts.begin(), pts.end(), [](const SiftPoint &a, const SiftPoint &b) {
    if (a.xpos!=b.xpos) return a.xpos<b.xpos;
    if (a.ypos!=b.ypos) return a.ypos<b.ypos;
    if (a.scale!=b.scale) return a.scale<b.scale;
    return a.orientation<b.orientation;
  });
  return pts;
}

// Same points, with the same descriptors and (if withMatches) the same matches
static bool Identical(SiftData &a, SiftData &b, bool withMatches)
{
  if (a.numPts!=b.numPts)
    return false;
  std::vector<SiftPoint> pa = Sorted(a), pb = Sorted(b);
  for (int i=0;i<a.numPts;i++) {
    const SiftPoint &p = pa[i], &q = pb[i];
    if (p.xpos!=q.xpos || p.ypos!=q.ypos || p.scale!=q.scale || p.sharpness!=q.sharpness ||
        p.edgeness!=q.edgeness || p.orientation!=q.orientation || memcmp(p.data, q.data, sizeof(p.data)))
      return false;
    if (withMatches && (p.score!=q.score || p.ambiguity!=q.ambiguity || p.match_xpos!=q.match_xpos || p.match_ypos!=q.match_ypos))
      return false;
  }
  return true;
}

static bool Load(CudaImage &img, cv::Mat &fimg, const char *file, int w = 0, int h = 0)
{
  cv::Mat src = cv::imread(file, 0), resized;
  if (src.empty())
    return false;
  if (w>0)
    cv::resize(src, resized, cv::Size(w, h));
  else
    resized = src;
  resized.convertTo(fimg, CV_32FC1);
  img.Allocate(fimg.cols, fimg.rows, iAlignUp(fimg.cols, 128), false, NULL, (float *)fimg.data);
  img.Download();
  return true;
}

// Largest distance between the projections by H1 and H2 of the points that are inliers of H1
static double MaxDistance(SiftData &data, const float *H1, const float *H2, float thresh)
{
  double maxd = 0.0;
  for (int i=0;i<data.numPts;i++) {
    const SiftPoint &p = data.h_data[i];
    float u[2], v[2];
    for (int k=0;k<2;k++) {
      const float *H = (k==0 ? H1 : H2);
      float z = H[6]*p.xpos + H[7]*p.ypos + H[8];
      u[k] = (H[0]*p.xpos + H[1]*p.ypos + H[2])/z;
      v[k] = (H[3]*p.xpos + H[4]*p.ypos + H[5])/z;
    }
    if (std::hypot(u[0] - p.match_xpos, v[0] - p.match_ypos)<thresh)
      maxd = std::max(maxd, (double)std::hypot(u[0] - u[1], v[0] - v[1]));
  }
  return maxd;
}

int main(int argc, char **argv)
{
  std::cout << "=============================================" << std::endl;
  std::cout << "  CudaSift Batch Tests" << std::endl;
  std::cout << "=============================================" << std::endl;
  InitCuda(argc>1 ? std::atoi(argv[1]) : 0);

  CudaImage img1, img2;
  cv::Mat f1, f2;
  CHECK(Load(img1, f1, "data/img1.png") && Load(img2, f2, "data/img2.png"), "Images loaded");
  SiftData a1, a2, b1, b2;
  InitSiftData(a1, 32768, true, true);
  InitSiftData(a2, 32768, true, true);
  InitSiftData(b1, 32768, true, true);
  InitSiftData(b2, 32768, true, true);

  std::cout << "\n--- Test: Extraction and matching in one batch ---" << std::endl;
  ExtractSift(a1, img1, 5, 1.0f, 3.0f);
  ExtractSift(a2, img2, 5, 1.0f, 3.0f);
  MatchSiftData(a1, a2);
  BeginSiftBatch();
  ExtractSift(b1, img1, 5, 1.0f, 3.0f);
  ExtractSift(b2, img2, 5, 1.0f, 3.0f);
  MatchSiftData(b1, b2);
  EndSiftBatch();
  double gpuTime = WaitSiftBatch();
  CHECK(b1.numPts>0 && b2.numPts>0, "Points extracted in the batch");
  CHECK(gpuTime>0.0, "WaitSiftBatch returns the GPU time");
  CHECK(Identical(a2, b2, false), "Batched extraction identical to immediate extraction");
  CHECK(Identical(a1, b1, true), "Batched matching identical to immediate matching");

  std::cout << "\n--- Test: Homography estimation in a batch ---" << std::endl;
  {
    float H1[9], H2[9];
    int n1, n2, fit1, fit2;
    FindHomographyGPU(a1, H1, &n1, 10000, 0.00f, 0.80f, 5.0f, 3);
    ImproveHomographyGPU(a1, H1, &fit1, 5, 0.00f, 0.80f, 3.0f);
    BeginSiftBatch();
    FindHomographyGPU(a1, H2, &n2, 10000, 0.00f, 0.80f, 5.0f, 3);
    ImproveHomographyGPU(a1, H2, &fit2, 5, 0.00f, 0.80f, 3.0f);
    EndSiftBatch();
    WaitSiftBatch();
    CHECK(n1==n2 && fit1==fit2 && !memcmp(H1, H2, sizeof(H1)), "Batched homography estimation identical");
    // Everything in one batch. The point order differs from a1, so the refinement sums
    // are rounded differently.
    BeginSiftBatch();
    ExtractSift(b1, img1, 5, 1.0f, 3.0f);
    ExtractSift(b2, img2, 5, 1.0f, 3.0f);
    MatchSiftData(b1, b2);
    FindHomographyGPU(b1, H2, &n2, 10000, 0.00f, 0.80f, 5.0f, 3);
    ImproveHomographyGPU(b1, H2, &fit2, 5, 0.00f, 0.80f, 3.0f);
    EndSiftBatch();
    WaitSiftBatch();
    double maxd = MaxDistance(a1, H1, H2, 3.0f);
    std::cout << "    " << n2 << " matches, " << fit2 << " inliers (immediate: " << n1 << ", " << fit1 << "), max distance " << maxd << " px" << std::endl;
    CHECK(n1==n2 && fit1==fit2 && maxd<0.001, "Extraction, matching and homography in one batch: same result");
  }

  std::cout << "\n--- Test: Upscaled extraction in a batch ---" << std::endl;
  ExtractSift(a1, img1, 5, 1.0f, 3.0f, 0.0f, true);
  BeginSiftBatch();
  ExtractSift(b1, img1, 5, 1.0f, 3.0f, 0.0f, true);
  EndSiftBatch();
  WaitSiftBatch();
  CHECK(Identical(a1, b1, false), "Batched upscaled extraction identical");

  std::cout << "\n--- Test: Point counts set on the host ---" << std::endl;
  ExtractSift(a1, img1, 5, 1.0f, 3.0f);
  ExtractSift(a2, img2, 5, 1.0f, 3.0f);
  memcpy(b1.h_data, a1.h_data, sizeof(SiftPoint)*a1.numPts);   // same points in the same order
  memcpy(b2.h_data, a2.h_data, sizeof(SiftPoint)*a2.numPts);
  a1.numPts = b1.numPts = 1001;
  a2.numPts = b2.numPts = 37;
  MatchSiftData(a1, a2);
  BeginSiftBatch();
  MatchSiftData(b1, b2);
  EndSiftBatch();
  WaitSiftBatch();
  bool same = true;
  for (int i=0;i<1001;i++)
    same &= (a1.h_data[i].score==b1.h_data[i].score && a1.h_data[i].match==b1.h_data[i].match);
  CHECK(same, "Batched matching of a 1001 x 37 subset identical");

  std::cout << "\n--- Test: Frame loop (extract a 1080p frame, match it with the previous one) ---" << std::endl;
  CudaImage frames[2];
  cv::Mat ff[2];
  Load(frames[0], ff[0], "data/img1.png", 1920, 1080);
  Load(frames[1], ff[1], "data/img2.png", 1920, 1080);
  float *mem = AllocSiftTempMemory(1920, 1080, 5, false);
  SiftData *data[2] = {&a1, &a2};
  double median[2];
  for (int batched=0;batched<2;batched++) {
    ExtractSift(*data[0], frames[0], 5, 1.0f, 3.0f, 0.0f, false, mem);
    std::vector<double> times;
    for (int i=1;i<=200;i++) {
      SiftData &cur = *data[i%2], &prev = *data[(i+1)%2];
      auto t0 = std::chrono::steady_clock::now();
      if (batched)
        BeginSiftBatch();
      ExtractSift(cur, frames[i%2], 5, 1.0f, 3.0f, 0.0f, false, mem);
      MatchSiftData(prev, cur);
      if (batched) {
        EndSiftBatch();
        WaitSiftBatch();
      }
      times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    median[batched] = times[times.size()/2];
  }
  std::cout << "    Immediate: " << median[0] << " ms per frame, batched: " << median[1] << " ms per frame" << std::endl;
  FreeSiftTempMemory(mem);

  FreeSiftData(a1);
  FreeSiftData(a2);
  FreeSiftData(b1);
  FreeSiftData(b2);
  std::cout << "\n=============================================" << std::endl;
  std::cout << "  Results: " << testsPassed << " passed, " << testsFailed << " failed" << std::endl;
  std::cout << "=============================================" << std::endl;
  return testsFailed>0 ? 1 : 0;
}
