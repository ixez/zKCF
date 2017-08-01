#include "KCF.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>

int main(int argc, char* argv[])
{
    using namespace cv;
    int startFrm=atoi(argv[2]), endFrm=atoi(argv[3]);

	zkcf::KCF tracker;
	bool pause = false;
    for(int i=startFrm;i<endFrm;i++) {
        char path[500];
        sprintf(path, argv[1], i);
        Mat frm = imread(path);
        if (i == startFrm) {
            tracker.Init(frm, Rect(atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7])));
        } else {
            Rect res = tracker.update(frm);
            Mat render = frm.clone();
            rectangle(render, res, CV_RGB(0, 255, 0), 3);
            imshow("show", render);

            int key = waitKey(pause ? 0 : 1);
            if (key == 32) {
                pause = !pause;
            }
        }
    }
    return EXIT_SUCCESS;
}