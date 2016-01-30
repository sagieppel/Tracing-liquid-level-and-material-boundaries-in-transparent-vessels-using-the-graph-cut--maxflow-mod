

#ifndef ImageGeneral_h
#define ImageGeneral_h
#include <opencv2\core\core.hpp>




#include <cstdio>
#include <iostream>
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<opencv\cxcore.h>
#include <cstdio>
#include <iostream>
using namespace std;
using namespace cv;
//#pragma once

enum TypeOfImage {binary=1, grey=2, color=3, number=4, none=5};

class ImageGeneral
{
public:
	Mat Img;
	int Hight,Width;
	TypeOfImage ImageType;
	string winname;
	ImageGeneral(void): Hight(0),Width(0),ImageType(none), winname("Image") { Img=imread("C:\\Users\\mithycow\\Desktop\\trial pictures glassware\\edited\\moor cut\\DSC_0016.jpg");	UpdateParamters();winname="Image";};
	ImageGeneral(const char *filename)  { Img=imread(filename);	UpdateParamters();winname="Image";};
	ImageGeneral(Mat &im)  { im.copyTo(Img);	 UpdateParamters();winname="Image";};
	ImageGeneral(ImageGeneral &im)  { Img=im.Img.clone();	 UpdateParamters();winname="Image"; };
	void Display(){namedWindow(winname);imshow(winname,Img);};
	void UpdateParamters()
	   {
		Hight=Img.rows;Width=Img.cols;
		if (Img.channels()==1) ImageType=grey;else if (Img.channels()==3) ImageType=color;
	   };
	Vec3b GetAtColor(int y, int x) const { if (ImageType==color) return Img.at<Vec3b>(y,x); };
	uchar GetAtColor(int y, int x,int ch) const { if (ImageType==color) return Img.at<Vec3b>(y,x)[ch]; }// same as before only with return specific rgb channel
	uchar GetAt(int y, int x) { if (ImageType==grey) return Img.at<uchar>(y,x); };
	uchar* At(int y, int x) { if (ImageType==grey) return &Img.at<uchar>(y,x); };
	void SetAtColor(Vec3b &v,int y, int x) { Img.at<Vec3b>(y,x)=v; };
	void SetAtColor(uchar v, int y, int x,int ch) {  Img.at<Vec3b>(y,x)[ch]=v; };// same as before only with return specific rgb channel
	void SetAt(int y, int x) const { Img.at<uchar>(y,x); };
	void ConvertToGrey() {Mat yyy;cvtColor(Img, yyy ,CV_RGB2GRAY);Img=yyy; ImageType=grey;}//Mat u; Img.convertTo(u,8U); Img=u; ImageType=grey;}
	void ConvertToBw() {Mat yyy;cvtColor(Img, yyy ,CV_RGB2GRAY);Img=yyy; ImageType=grey;}
	void Resize(int sy,int sx=0)
	{ if (sx==0) sx=int(float(Width*sy)/(Hight));
	resize(Img,Img ,Size(sx,sy));
	Hight=sy; Width=sx;
	}
	void CopyTo(ImageGeneral &im){ im.Img=Img.clone();im.UpdateParamters();};
	void CopyFrom(ImageGeneral &im){ Img=im.Img.clone();UpdateParamters();};
	void CopyTo(Mat &im){ im=Img.clone();};
	void CopyFrom(Mat &im){ Img=im.clone();};
	void operator =(ImageGeneral &im) { Img=im.Img;UpdateParamters();};
	void operator =(Mat &im) { Img=im;UpdateParamters();};
	void operator *=(ImageGeneral &im) { Img*=im.Img; UpdateParamters();};
	void operator *=(double d) { Img*=d; UpdateParamters();};
	Mat operator *(double d) {return Img*d; };
	Mat operator +(Mat &m) {return Img+m; };
	uchar*  operator()(int y, int x)  { if (ImageType==grey) return &Img.at<uchar>(y,x); };
	
	Mat  Crop(int y1,int x1,int y2, int x2) const {return Img(Rect(y1,x1,y2,x2));};

	~ImageGeneral(void){};
};
#endif
