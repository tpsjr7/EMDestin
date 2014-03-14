

#include <string>
#include <fstream>
#include "stdio.h"
#include "VideoSource.h"
#include "DestinNetworkAlt.h"
#include "Transporter.h"
#include "unit_test.h"
#include <time.h>
#include "macros.h"
#include "CifarSource.h"
#include "stereovision.h"
#include "stereocamera.h"
#include "ImageSourceImpl.h"
#include "CztMod.h"
#include "BeliefExporter.h"
#include "SomPresentor.h"
#include "ClusterSom.h"
#include <stdlib.h>
#include <vector>

#define NUMBER "0"
using namespace std;
using namespace cv;
vector<Mat> spl;
ImageSouceImpl isi;

int main(int argc, char ** argv) {


    /*-----------------------------------------initialization-------------------------------------------------*/
    uint centroid_counts[] = {16, 8, 8, 8};
    uint numLayers = 4;
    bool isUniform = true;
    int size = 32 * 32;
    SupportedImageWidths siw = W32;
    DestinNetworkAlt dn(siw, numLayers, centroid_counts, isUniform, 1);
    string CifarDir = "/home/opencog/Downloads/cifar-10-batches-bin";
    int TrainBatch = 1;
    int TestBatch = 2;
    int frameCount = 0;
    CifarSource csTrain(CifarDir, TrainBatch);
    CifarSource csTest(CifarDir, TestBatch);
    dn.setParentBeliefDamping(0);
    dn.setPreviousBeliefDamping(0);
    int maxCount = 10000;
    int bottom_belief_layer = 0;
    /*----------------------------------------------training---------------------------------------*/
    BeliefExporter be(dn, bottom_belief_layer);
    int iterations_per_image = 8;
    //int layers=4;
    bool isEnabled = true;
    csTrain.disableAllClasses();
    csTrain.setClassIsEnabled(0, isEnabled);
    csTrain.setClassIsEnabled(1, isEnabled);
    csTrain.setClassIsEnabled(2, isEnabled);
    csTrain.setClassIsEnabled(3, isEnabled);
    csTrain.setClassIsEnabled(4, isEnabled);
    csTrain.setClassIsEnabled(5, isEnabled);
    csTrain.setClassIsEnabled(6, isEnabled);
    csTrain.setClassIsEnabled(7, isEnabled);
    csTrain.setClassIsEnabled(8, isEnabled);
    csTrain.setClassIsEnabled(9, isEnabled);

    for (int i = 0; i < maxCount; ++i) {
        //while(i<3){
        //if (i%200==0)
        //printf("Iteration Number: %d \n",i);

        csTrain.findNextImage();
        dn.clearBeliefs();
        for (int j = 0; j < numLayers; ++j) {
            dn.setLayerIsTraining(j, false); //now
        }
        for (int j = 0; j < numLayers; ++j) {
            dn.setLayerIsTraining(j, true);
            dn.doDestin(csTrain.getGrayImageFloat());
        }
        //string FileName="TestBeliefs.txt";
        //be.CreateFile(FileName);
        //be.DumpBeliefs(FileName);
        for (int j = 0; j < 2; ++j) {
            dn.setLayerIsTraining(j, true);
            dn.doDestin(csTrain.getGrayImageFloat());
        }
        i++;

    }

    /*-----------------------------------------------Testing/Dumping Beliefs to a File  ---------------------------------------------------*/
    for (int j = 0; j < numLayers; j++) {
        dn.setLayerIsTraining(j, false);
    }
    string FileName = "TestBeliefs.txt";
    be.CreateFile(FileName);
    for (int i = 1; i < maxCount; ++i) {
        csTest.setCurrentImage(i);
        dn.clearBeliefs();
        for (int j = 0; j < numLayers; ++j) {
            dn.doDestin(csTest.getGrayImageFloat());
        }

        be.DumpBeliefs(FileName);

    }


    return 0;
}