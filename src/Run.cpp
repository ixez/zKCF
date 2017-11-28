#include "KCF.h"
#include "TrackTask.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    using namespace cv;
    using namespace zkcf;
    using namespace std;
    ztrack::TrackTask conf;
    conf.SetArgs(argc, argv);

    Mat frm;
    Rect result;

    KCF tracker;
//    KCF tracker(FeatureType::FEAT_RAW);

    for (int frameId = conf.StartFrmId, i = 1; frameId <= conf.EndFrmId; ++frameId, ++i)
    {
        // Read each frame from the list
        frm = conf.GetFrm(frameId);
        if (i == 1) {
            result=conf.Bbox;
            tracker.Init(frm, result);
        } else {
            result = tracker.Track(frm);
        }
        conf.PushResult(result);
    }
    conf.SaveResults();
    return EXIT_SUCCESS;
}