
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
    for(int i=1;i<5;i++){
        char name[256];
        sprintf(name, "/home/opencog/Desktop/Min/0/0/%d.png", i);
        isi.addImage(name);
    }
/*-----------------------------------------initialization-------------------------------------------------*/
        SupportedImageWidths siw = W32;
        uint centroid_counts[]  = {16,8,4,1};
        uint numLayers=4;
        bool isUniform = true;
        int size = 32*32;
        //int extRatio = 1;
        DestinNetworkAlt * dn = new DestinNetworkAlt(siw, numLayers, centroid_counts, isUniform, 1);
        string CifarDir="/Downloads/cifar-10-batches-bin";
        int TrainBatch=1;
        int TestBatch=2;
        int frameCount = 0;
        //clock_t start,finish;
        csTrain=CifarSource(CifarDir,TrainBatch);
        csTest=CifarSource(CifarDir,TestBatch);
        dn.setParentBeliefDamping(0);
        dn.setPreviousBeliefDamping(0);
        int maxCount =10000;
/*----------------------------------------------training---------------------------------------*/
#if 1
    Node * n = network->getNode(2, 0, 0);
        while(frameCount < maxCount){
            frameCount++;
             isi.findNextImage();
        network->doDestin(isi.getGrayImageFloat());
        if(frameCount%10==0){
        printf("frameCount%d\n",frameCount);

        }

    }

    network->save("treeminer.dst");
    #else
    network->load("treeminer.dst");
    #endif
/*-----------------------------------------------test---------------------------------------------------*/
    //  Destin * dn = network->getNetwork();
    for(int i=1;i<10;i++){
        stringstream ss;
        string str;
        ss<<i+1;
        ss>>str;
        string name2="/home/opencog/Desktop/Min/0/0/"+str+".png";
        ImageSouceImpl test;
        test.addImage(name2);
        for(int i=0;i<10;i++)
        {
            test.findNextImage();
            network->TestDestin(test.getGrayImageFloat());
        }
        char name1[256];
        sprintf(name1, "/home/opencog/Desktop/Min/0/1/destin_%d.txt", frameCount);
        ofstream outfile(name1,ofstream::out|ofstream::app);
        if(!outfile)
        {
            cerr<<"open train.txt erro!"<<endl;
            exit(1);
        }
        Node * n = network->getNode(2, 0, 0);
        std::stringstream ss1;
        std::string str1;
        outfile<<";;"<<endl;
    for(int i = 0; i < n->nb ; i++){
            outfile<<n->pBelief[i]<<",";
        printf("%f ", n->pBelief[i]);
        }
    n = network->getNode(2, 0, 1);
    for(int i = 0; i < n->nb ; i++){
            outfile<<n->pBelief[i]<<",";
        printf("%f ", n->pBelief[i]);
        }
    n = network->getNode(2,1, 0);
    for(int i = 0; i < n->nb ; i++){
            outfile<<n->pBelief[i]<<",";
        printf("%f ", n->pBelief[i]);
        }
    n = network->getNode(2, 1, 1);
    for(int i = 0; i < n->nb ; i++){
            outfile<<n->pBelief[i]<<",";
        printf("%f ", n->pBelief[i]);
        }

    //outfile.put(network->printNodeBeliefs(7,0,0));
    outfile.close();
        }

    printf("Here We come to the end \n");
    return 0;

}
