/*
Copyright (c) 2016, TU Dresden
Copyright (c) 2017, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <iostream>
#include <fstream>

#include "properties.h"
#include "thread_rand.h"
#include "util.h"
#include "stop_watch.h"
#include "dataset.h"

#include "lua_calls.h"
#include "cnn.h"

#include <iostream>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#define AXES_LEN 2 //坐标轴长
double rrx = 0;
double rry = 0;
double rrz = 0;
double ttx = 0;
double tty = 0;
double ttz = 0;
double rrxGT = 0;
double rryGT = 0;
double rrzGT = 0;
double ttxGT = 0;
double ttyGT = 0;
double ttzGT = 0;
unsigned i = 0;
double avgCorrect = 0;
std::vector<double> expLosses;
std::vector<double> sfEntropies;
std::vector<double> rotErrs;
std::vector<double> tErrs;
jp::Dataset testDataset = jp::Dataset("./test/");
lua_State* stateRGB = luaL_newstate();
lua_State* stateObj;
int objHyps;
cv::Mat camMat;
int inlierThreshold2D;
int refSteps;
GlobalProperties* gp= GlobalProperties::getInstance();
std::ofstream testFile;
std::ofstream testErrFile;

void init ()
{
    glClearColor(0.0,0.0,0.0,0.0);
    glShadeModel(GL_FLAT);
}


void display()
{

    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    GLUquadricObj *quadric =gluNewQuadric();
    glPushMatrix();
    gluLookAt(1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    glPushMatrix();
    glColor3f(1.0,1.0, 1.0);

    glutSolidSphere(0.05,6,6);
    glColor3f(0.0,0.0, 1.0);
    gluCylinder(quadric,0.01, 0.01, AXES_LEN, 10, 5);         //Z
    glTranslatef(0,0,AXES_LEN);
    gluCylinder(quadric,0.03, 0.0, 0.06, 10, 5);                 //Z
    glPopMatrix();
    glPushMatrix();
    glTranslatef(0,0.5,AXES_LEN);
    glRotatef(90,0.0,1.0,0.0);
    //GLPrint("Z")  ;                                             //Print GL Text ToThe Screen
    glPopMatrix();
    glPushMatrix();

    glColor3f(0.0,1.0, 0.0);
    glRotatef(-90,1.0,0.0,0.0);
    gluCylinder(quadric,0.01, 0.01, AXES_LEN, 10, 5);         //Y
    glTranslatef(0,0,AXES_LEN);
    gluCylinder(quadric,0.03, 0.0, 0.06, 10, 5);                 //Y
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0.5,AXES_LEN,0);
    //GLPrint("Y");                                               //Print GL Text ToThe Screen
    glPopMatrix();

    glPushMatrix();
    glColor3f(1.0,0.0, 0.0);
    glRotatef(90,0.0,1.0,0.0);
    gluCylinder(quadric,0.01, 0.01, AXES_LEN, 10, 5);        //X
    glTranslatef(0,0,AXES_LEN);
    gluCylinder(quadric,0.03, 0.0, 0.06, 10, 5) ;               //X
    glPopMatrix();

    glPushMatrix();
    glTranslatef(AXES_LEN,0.5,0);
    //GLPrint("X")   ;                                           // # Print GL Text ToThe Screen
    glPopMatrix();

    glRotatef(rrz/3.14*360,0.0,0.0,1.0);
    glRotatef(rry/3.14*360,0.0,1.0,0.0);
    glRotatef(rrx/3.14*360,1.0,0.0,0.0);
    

    glTranslatef(ttx,tty,ttz);
    glPushMatrix();
    glScalef(0.2,0.2,0.1);
    glutWireCube(1.0);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0,0,-0.15);
    gluCylinder(quadric,0.07, 0.05, 0.1,20,3);
    glPopMatrix();

    glColor3f(0.0,1.0, 0.0);
    glRotatef(rrzGT/3.14*360,0.0,0.0,1.0);
    glRotatef(rryGT/3.14*360,0.0,1.0,0.0);
    glRotatef(rrxGT/3.14*360,1.0,0.0,0.0);
    
    glTranslatef(ttxGT,ttyGT,ttzGT);

    glPushMatrix();
    glScalef(0.1,0.1,0.05);
    glutWireCube(1.0);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0,0,-0.078);
    gluCylinder(quadric,0.035, 0.025, 0.05,20,3);
    glPopMatrix();


    glPopMatrix();
    glFlush();
    glutSwapBuffers();
}

void reshape(int w,int h)
{
    glViewport(0,0,(GLsizei)w,(GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(30,(GLfloat) w/ (GLfloat)h,5,100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0,0.0,-6.0);
}

void trchange()

{       
    

        std::cout << YELLOWTEXT("Processing test image " << i+1<< " of " << testDataset.size()) << "." << std::endl;

        // load test image
        jp::img_bgr_t testRGB;
        testDataset.getBGR(i, testRGB);

        jp::cv_trans_t hypGT;
        testDataset.getPose(i, hypGT);

        std::cout << BLUETEXT("Predicting object coordinates.") << std::endl;

        cv::Mat_<cv::Point2i> sampling;
        std::vector<cv::Mat_<cv::Vec3f>> imgMaps;
        jp::img_coord_t estObj = getCoordImg(testRGB, sampling, imgMaps, false, stateRGB);
	cv::imshow("testrgb",testRGB);
	cv::waitKey(10);
        // process frame (same function used in training, hence most of the variables below are not used here), see method documentation for parameter explanation
        std::vector<jp::cv_trans_t> refHyps;
        std::vector<double> sfScores;
        std::vector<std::vector<cv::Point2i>> sampledPoints;
        std::vector<double> losses;
        std::vector<cv::Mat_<int>> inlierMaps;
        double tErr;
        double rotErr;
        int hypIdx;

        double expectedLoss;
        double sfEntropy;
        bool correct;

        processImage(
            hypGT,
            stateObj,
            objHyps,
            camMat,
            inlierThreshold2D,
            refSteps,
            expectedLoss,
            sfEntropy,
            correct,
            refHyps,
            sfScores,
            estObj,
            sampling,
            sampledPoints,
            losses,
            inlierMaps,
            tErr,
            rotErr,
            hypIdx,
            false);
	avgCorrect += correct;
        // invert pose to get camera pose (we estimated the scene pose)
        jp::cv_trans_t invHyp = getInvHyp(refHyps[hypIdx]);
	//show 3D model of rgb pic
	//cv::imshow("estobj",invHyp);
	//cv::waitKey(100); 
        testErrFile
            << expectedLoss << " "      // 0  - expected loss over the hypothesis pool
            << sfEntropy << " "         // 1  - entropy of the hypothesis score distribution
            << losses[hypIdx] << " "    // 2  - loss of the selected hypothesis
            << tErr << " "              // 3  - translational error in m
            << rotErr << " "            // 4  - rotational error in deg
            << invHyp.first.at<double>(0, 0) << " "     // 5  - selected pose, rotation (1st component of Rodriguez vector)
            << invHyp.first.at<double>(1, 0) << " "     // 6  - selected pose, rotation (2nd component of Rodriguez vector)
            << invHyp.first.at<double>(2, 0) << " "     // 7  - selected pose, rotation (3th component of Rodriguez vector)
            << invHyp.second.at<double>(0, 0) << " "    // 8  - selected pose, translation in m (x)
            << invHyp.second.at<double>(0, 1) << " "    // 9  - selected pose, translation in m (y)
            << invHyp.second.at<double>(0, 2) << " "    // 10 - selected pose, translation in m (z)
            << std::endl;
        rrx=invHyp.first.at<double>(0, 0);
        rry=invHyp.first.at<double>(1, 0);
        rrz=invHyp.first.at<double>(2, 0);
	ttx=invHyp.second.at<double>(0, 0);
	tty=invHyp.second.at<double>(0, 1);
	ttz=invHyp.second.at<double>(0, 2);
        std::cout<<YELLOWTEXT("the test image " << i+1 << " of " << testDataset.size()<<":" ) << std::endl;
        std::cout<< YELLOWTEXT("Pose rotation:")<<std::endl;
	std::cout<<"x:"<<invHyp.first.at<double>(0, 0)<<std::endl;
	std::cout<<"y:"<<invHyp.first.at<double>(1, 0)<<std::endl;
	std::cout<<"z:"<<invHyp.first.at<double>(2, 0)<<std::endl;
	std::cout<<"xG:"<<hypGT.first.at<double>(0, 0)<<std::endl;
	std::cout<<"yG:"<<hypGT.first.at<double>(1, 0)<<std::endl;
	std::cout<<"zG:"<<hypGT.first.at<double>(2, 0)<<std::endl;
        std::cout<< YELLOWTEXT("Pose translation:")<<std::endl;
	std::cout<<"x:"<<invHyp.second.at<double>(0, 0)<<std::endl;
	std::cout<<"y:"<<invHyp.second.at<double>(0, 1)<<std::endl;
	std::cout<<"z:"<<invHyp.second.at<double>(0, 2)<<std::endl;
	std::cout<<"xG:"<<hypGT.second.at<double>(0, 0)<<std::endl;
	std::cout<<"yG:"<<hypGT.second.at<double>(0, 1)<<std::endl;
	std::cout<<"zG:"<<hypGT.second.at<double>(0, 2)<<std::endl;
        


        expLosses.push_back(expectedLoss);
        sfEntropies.push_back(sfEntropy);
        tErrs.push_back(tErr);
        rotErrs.push_back(rotErr);

   	
    

	display();
   if(i == testDataset.size()-1){
    // mean and stddev of loss
    std::vector<double> lossMean;
    std::vector<double> lossStdDev;
    cv::meanStdDev(expLosses, lossMean, lossStdDev);
    
    // mean and stddev of score entropy
    std::vector<double> entropyMean;
    std::vector<double> entropyStdDev;
    cv::meanStdDev(sfEntropies, entropyMean, entropyStdDev);
	
    avgCorrect /= testDataset.size() / gp->dP.imageSubSample;
    
    // median of rotational and translational errors
    std::sort(rotErrs.begin(), rotErrs.end());
    std::sort(tErrs.begin(), tErrs.end());
    
    double medianRotErr = rotErrs[rotErrs.size() / 2];
    double medianTErr = tErrs[tErrs.size() / 2];
    
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << BLUETEXT("Avg. test loss: " << lossMean[0] << ", accuracy: " << avgCorrect * 100 << "%") << std::endl;
    std::cout << "Median Rot. Error: " << medianRotErr << "deg, Median T. Error: " << medianTErr * 100 << "cm." << std::endl;

    testFile
            << avgCorrect << " "            // 0 - percentage of correct poses
            << lossMean[0] << " "           // 1 - mean loss of selected hypotheses
            << lossStdDev[0] << " "         // 2 - standard deviation of losses of selected hypotheses
            << entropyMean[0] << " "        // 3 - mean of the score distribution entropy
            << entropyStdDev[0] << " "      // 4 - standard deviation of the score distribution entropy
            << medianRotErr << " "          // 5 - median rotational error of selected hypotheses
            << medianTErr                   // 6 - median translational error (in m) of selected hypotheses
            << std::endl;

         testFile.close();
         testErrFile.close();
	 lua_close(stateRGB);
         lua_close(stateObj);
	 exit(1);}
    if(i <  testDataset.size()-1){
        i+= gp->dP.imageSubSample;}
        
        


	    
}

int main(int argc,const char *argv[])
{ 
    // read parameters
    
    gp->parseConfig();
    gp->parseCmdLine(argc, argv);
    //gp->tP.objScript="train_obj.lua";    //-oscript
    //gp->tP.objModel="obj_model_fcn_e2e.net"; //-omodel
    //gp->tP.scoreScript="score_incount_ec6.lua";  //-sscript
    //gp->tP.randomDraw=0;  //-rdraw
    

    objHyps = gp->tP.ransacIterations;
    inlierThreshold2D = gp->tP.ransacInlierThreshold;
    refSteps = gp->tP.ransacRefinementIterations;  
    
    std::string baseScriptRGB = gp->tP.objScript;
    std::string baseScriptObj = gp->tP.scoreScript;
    std::string modelFileRGB = gp->tP.objModel;

    // setup data and torch
    std::cout << std::endl << BLUETEXT("Loading test set ...") << std::endl;

    // lua and models
    std::cout << "Loading script: " << baseScriptObj << std::endl;
    stateObj = luaL_newstate();
    luaL_openlibs(stateObj);
    execute(baseScriptObj.c_str(), stateObj);
    loadScore(inlierThreshold2D, gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateObj);
    
    std::cout << "Loading script: " << baseScriptRGB << std::endl;
    stateRGB = luaL_newstate();
    luaL_openlibs(stateRGB);
    execute(baseScriptRGB.c_str(), stateRGB);
    loadModel(modelFileRGB, gp->getCNNInputDimX(), gp->getCNNInputDimY(), gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateRGB);
       
    camMat = gp->getCamMat();
    
    testFile.open("ransac_test_loss_"+baseScriptRGB+"_rdraw"+intToString(gp->tP.randomDraw)+"_"+gp->tP.sessionString+".txt"); // contains evaluation information for the whole test sequence
    testErrFile.open("ransac_test_errors_"+baseScriptRGB+"_rdraw"+intToString(gp->tP.randomDraw)+"_"+gp->tP.sessionString+".txt"); // contains evaluation information for each test image
    setEvaluate(stateRGB);
    setEvaluate(stateObj);

    int aragc=1;
    char *aragv[1]={(char*)"something"};
    glutInit(&aragc, aragv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition(900, 100);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Camera pose");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(trchange);
    
    glutMainLoop();
    return 0;

}




