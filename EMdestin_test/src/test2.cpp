
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

int main(int argc, char ** argv)
{
 /*   for(int i=1;i<5;i++){
        char name[256];
        sprintf(name, "/home/opencog/Desktop/Min/0/0/%d.png", i);
        isi.addImage(name);
   } */
/*-----------------------------------------initialization-------------------------------------------------*/
        uint centroid_counts[]  = {16,8,8,8};
        uint numLayers=4;
        bool isUniform = true;
        int size = 32*32;
        SupportedImageWidths siw= W32;
        //int extRatio = 1;
        DestinNetworkAlt dn(siw, numLayers, centroid_counts, isUniform, 1);
        string CifarDir="/home/opencog/Downloads/cifar-10-batches-bin";
        // string CifarDir="/Downloads/cifar-10-batches-bin";
        int TrainBatch=1;
        int TestBatch=2;
        int frameCount = 0;
        //clock_t start,finish;
        CifarSource csTrain(CifarDir,TrainBatch);
        CifarSource csTest(CifarDir,TestBatch);
        dn.setParentBeliefDamping(0);
        dn.setPreviousBeliefDamping(0);
        int maxCount =10000;
        int bottom_belief_layer=0;

/*----------------------------------------------training---------------------------------------*/
    BeliefExporter be(dn,bottom_belief_layer);
    int iterations_per_image=8;
    //int layers=4;
    bool isEnabled=true;
    csTrain.disableAllClasses();
    csTrain.setClassIsEnabled(0,isEnabled);
    csTrain.setClassIsEnabled(1,isEnabled);
    csTrain.setClassIsEnabled(2,isEnabled);
    csTrain.setClassIsEnabled(3,isEnabled);
    csTrain.setClassIsEnabled(4,isEnabled);
    csTrain.setClassIsEnabled(5,isEnabled);
    csTrain.setClassIsEnabled(6,isEnabled);
    csTrain.setClassIsEnabled(7,isEnabled);
    csTrain.setClassIsEnabled(8,isEnabled);
    csTrain.setClassIsEnabled(9,isEnabled);
   // csTrain.setClassIsEnabled();
   // csTest.setClassIsEnabled();
    //for(int i=0;i<maxCount;++i){
    for(int i=0;i<maxCount;++i){
        if (i%100==0){
            printf("Iteration Number %d \n",i);
        }
        csTrain.findNextImage();
        dn.clearBeliefs();
        for(int j=0;j<numLayers;++j){
            dn.setLayerIsTraining(j,false);
            }
        for(int j=0;j<numLayers;++j){
            dn.setLayerIsTraining(j,true);
            dn.doDestin(csTrain.getGrayImageFloat());
            }
        for(int j=0;j<2;++j){
            //dn.setLayerIsTraining(j,true);
            dn.doDestin(csTrain.getGrayImageFloat());
            }

    }


/*-----------------------------------------------Testing/Dumping Beliefs to a File  ---------------------------------------------------*/
    //  Destin * dn = network->getNetwork();
    for(int j=0;j<numLayers;j++){
        dn.setLayerIsTraining(j,false);
        }
    string FileName="TestBeliefs";
    be.CreateFile(FileName);
    for(int i=1;i<maxCount;++i){
        // Show DestinImage(i)
        csTest.setCurrentImage(i);
        dn.clearBeliefs();
        for(int j=0;j<numLayers;++j){
            dn.doDestin(csTest.getGrayImageFloat());
            }
        be.DumpBeliefs(FileName);

        }


    printf("Here We come to the end \n");
    return 0;

}
