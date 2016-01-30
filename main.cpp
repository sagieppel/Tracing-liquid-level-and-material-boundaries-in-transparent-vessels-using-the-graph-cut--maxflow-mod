// Code for graph cut based determination 
//#include "ImageGeneral.h" 
#include <opencv2\core\core.hpp>




#include <cstdio>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <cstdio>
#include <iostream>
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<opencv\cxcore.h>
#include <cstdio>
#include <iostream>
#include<math.h>
#include <windows.h>
#include "graph.h"// download to project directory
#include "dirent.h"


using namespace std;
using namespace cv;


double SQR(double x) {return(x*x);}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class point
{
public:
	unsigned int x,y;
	point(unsigned int xx,unsigned int yy){x=xx;y=yy;};
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class sides// for each line of the vessel the y value of the line and the left most and right most of x values of the vessel contour in this line
{
public:
	unsigned int x1,x2,y;
	sides(unsigned int yy,unsigned int xx1,unsigned int xx2){x2=xx2;x1=xx1;y=yy;};
};
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GraphCutDemo()// Demonstrate the use of the graph cut function (not used)
{
	typedef Graph<int,int,int> GraphType;
	GraphType *g = new GraphType(/*estimated number of nodes*/ 3, /*estimated number of edges*/ 3); 
	
	g -> add_node(); //add first node to the graph
	g -> add_node(); //add second node to the graph
	g -> add_node(); //add third node to the graph
	g -> add_tweights( 0,   /* capacities */  1, 15 );// first number is node number Second number is capcity(power of connection) to sink  and third is power of connection to source
	g -> add_tweights( 1,   /* capacities */  15, 1 );// first number is node number Second number is capcity(power of connection) to sink  and third is power of connection to source
	g -> add_edge( 0, 1,    /* capacities */  3, 3);//first and second numbers are the nodes that make the connection and third and forth numbers are the power of the connection between them (assuming MinSourceYmmetric connections the third and forth number should be identical)
		g -> add_edge( 0, 2,    /* capacities */  3, 3);//first and second numbers are the nodes that make the connection and third and forth numbers are the power of the connection between them (assuming symmetric connections the third and forth number should be identical)
			g -> add_edge( 1, 2,    /* capacities */  5, 5);//first and second numbers are the nodes that make the connection and third and forth numbers are the power of the connection between them (assuming symmetric connections the third and forth number should be identical)

	int flow = g -> maxflow();// calculate the graph cut for each node set value of either source or sink

	printf("Flow = %d\n", flow);
	printf("Minimum cut:\n");
	if (g->what_segment(0) == GraphType::SOURCE)// check for  node 0 if its a source or sink
		printf("node0 is in the SOURCE set\n");
	else
		printf("node0 is in the SINK set\n");
	if (g->what_segment(1) == GraphType::SOURCE)// check for  node 1 if its a source or sink
		printf("node1 is in the SOURCE set\n");
	else
		printf("node1 is in the SINK set\n");
	if (g->what_segment(2) == GraphType::SOURCE)// check for  node 1 if its a source or sink
		printf("node2 is in the SOURCE set\n");
	else
		printf("node2 is in the SINK set\n");
	
	delete g;
	int yyy;
	cin>>yyy;
	return 0;
}//not used demonstration code for using the graph cut class

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//********************************************************************Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************
//********************************************************************Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************
//********************************************************************Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************
//********************************************************************Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************
//********************************************************************Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Find_Phase_Boundary_By_Graph_Cut(Mat &Img, Mat &Ib,Mat &ImOut,  double Sigma=20, double k=0, 	bool UseAreaSinkSource=true,double SFract=0.15, double Margin=0.1,double KHorrizontalWeight=2, bool ShowMode=false, bool MixtureOfGaussian=false)
{
	// For image (Img) of material inside transparent vessel Find material boundary in the image and draw it on the ourput img (ImOut)
	// Use graph cut with the top of the vessel as source and bottum of the vessel as sink

	//Mat img is the Img of the tansparent vessel in with the material inside (color)
	
	//Ib binary image with of the vessel countour in Img, hence pixels in Img that correspond to the vessel boundary in Img are marked 1 in Ib the rest of the pixels marked zero
	//The contour in Ib is one pixel thick close contour line

	//ImOut the output image  with the phase boundary found marked on the image
	// double Sigma=1, Sigma for capicity weight caclaulaation  Sigma>0 mean use exponetial weight based on intensity difference between adjacent pixels.Sigma=0 mean use linear weight  based on intensity difference. Sigma=-1 mean use relative intensity as linear weight
//	double k;//The  weight of connection of each pixel to source and drain. How much the similarity of the pixel to the bottum or top of the vessel determine where the phase boundary pass. 
//  If this parameter (k) is zero the phase boundary (graph cut) is determined only by connection of pixels to neighbor pixels. When the material in the vessel have strong distinctive color k should high, if the material in the vessel is transperent (like water) k should be zero

	
//	bool UseAreaSinkSource=true;//The top and bottum of the vessel are marked as source and sink for graph cut. If this parameter is true use the area fracrtion (SFrac) in top and bottum of the vessel to define which pixel is sink or source
	//	double SFract=0.15;// define source and sink all pixels in the vessel region of the image  that appeaer in the top bottum SFract% of the vessel area in the image (use only if UseAreaSinkSource==true)

//	int Margin=0.1;// define source and sink all pixels in the vessel region of the image with  vertical distance to the vessel top or bottum that is lower then this parameter Margin*Vessel_Hight in pixels (use only if UseAreaSinkSource==false)
//	double KHorrizontalWeight=1 additional weight to be added to horzontal weight/capcity the capicity of all  horizontal connections betweeen pixels will be multiply by this value. This is used to discourage vertical and stip cuts
	//	bool ShowMode=true; //Do You whant to display proccess and transformation on screen? (mainly for debug mode)
	//bool MixtureOfGaussian=true if true Caclulate connection between each pixel to source and sink using mixture of gaussian model if false Calculate every pixel connection  to source and sink according to the difference between the pixel intensity and the rest of the pixels in the source and sink


	if (Ib.empty()) Ib=imread("C:\\image\\Ibor.tif");// boundaries of the vessel in the image
	if (Ib.empty()) {cout<<"Error Cant open Boundary image Ib";exit(0);}
	if (Img.empty())	{Img=imread("C:\\image\\I.jpg");}
	if (Img.empty()) {cout<<"Error Cant open image Img";exit(0);}	
			resize(Img,Img,Ib.size());
			//I.convertTo(I,
cvtColor(Ib, Ib ,CV_RGB2GRAY);//**********check this one out 
		cv::cvtColor(Img,Img,CV_RGB2GRAY);
		if (ShowMode)
		{
	            
	              imshow("Borders",Ib);
	              cv::imshow("Grey Image",Img);
                  cv::waitKey();
		}
	vector<int> SinkHistogram(256,0);//Histogram of sink pixels intensity
	vector<int> SourceHistogram(256,0);//Histogram of source pixels intensity
	long nSink=0,nSource=0;//number of pixels in source and sink
vector <point> TopC,LowC,Tl,Bl;
vector <sides> Sides;// array contain the leftmost and rightmost  pixels of every line of of the image
unsigned int MaxY=0,MinY=Ib.rows; // Y values of top and bottum lines of the vessel in the image  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	

///////////////////////////////////Find the contour of the vessel from the border image and put it in array containing the x coordinates right most and left most pixels of eachline in the vessel//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int maxx,minx;
long MeanWidth=0;//Mean width of the vessel in pixels
int MaxWidth=0;//Max width of the vessel in pixels
bool l=false;
for (int y=0;y<Ib.rows;y++)// go over all pixels in the image
{
	for (int x=0;x<Ib.cols;x++)		
	{
		if (Ib.at<uchar>(y,x)!=0)
		{
			if (l) maxx=x;
			else
		    {
			   l=1;
			   minx=x;
			   maxx=x;
		    }
		}
	}
	if (l)
	{
	   Sides.push_back(sides(y,minx,maxx));
       if (y>MaxY) MaxY=y;// Y value of top line of the vessel in the image
	   if (y<MinY) MinY=y;// Y value of bottum line of the vessel in the image
	   l=false;
	   MeanWidth+=abs(maxx-minx);
	   MaxWidth=max(MaxWidth,(maxx-minx));
	}


}
MeanWidth/= Sides.size();

/////////////////////////////////////////////Optional Create dilated version of the vessel boundary as the vessel boundary to be used as penalty zone for graph propogation (capcity in this region will be triple). This is optional and use to prevent phase booundary propogation along the vessel boundaries////////////////////////////////////////////
double PenaltyZoneWidth=0.1;//fraction of the vessel width that will be used for penalty zone Set for zero if you want to cancel the penalty zone
double Penalty=30; //the penalty for connection in the penalty zone
bool PenaltyZoneOn=(Penalty!=1 && PenaltyZoneWidth>0);//if true Use penalty zone
int DilationRadius=MAX(MeanWidth*PenaltyZoneWidth,1);//MaxWidth/5;//
Mat PenaltyZone;// Mat binary image of the penalty zone pixels capcity of horizontal connections in of pixels that marked 1 in this matrix will be multiply  by Penalty
if (PenaltyZoneOn)
{
  Mat element = getStructuringElement(MORPH_ELLIPSE, Size(DilationRadius, DilationRadius ));
  /// Apply the dilation operation
  dilate(Ib, PenaltyZone, element );
  if (ShowMode)
  {
  imshow( "Penalty Zone", PenaltyZone );
      cv::waitKey();
  }
} 
else PenaltyZone=Ib;
////////////////////////////////////////////Find and index vessel interior/////////////////////////////////////////////////////////////////////////////
//For  each pixel inside the vessel contour give an unique index number the index number of pixel in location x,y is put MInd.at<int>(y,x)
Mat MInd(Ib.size(), CV_32SC1, Scalar::all(-1));//mat with the index number of each cell
//	 vector <cv::Point> Vind; // array wiht indexes each part contain the point correspond to this index
	 int indx=-1;//the index of corrent pixel
	 int NumberOfNodes=0;
	 for (int f=1;f<Sides.size();f++)
		 for (int x=Sides[f].x1+1;x<Sides[f].x2;x++)
		 {
			 MInd.at<int>(Sides[f].y,x)=++indx;
			 NumberOfNodes++;
			// Vind.push_back(cv::Point(x,Sides[f].y));			
		 }
	if (ShowMode)	 imshow("index",MInd);
/////////////////////////////////initialize graph/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		typedef Graph<int,int,int> GraphType;
	GraphType *g = new GraphType(/*estimated number of nodes*/ NumberOfNodes, /*estimated number of edges*/ NumberOfNodes*2); 
	for (int f=0;f< NumberOfNodes;f++) g -> add_node();
////////////////////////////////////////////TOP AND BOTTOM MARGINS BY DISTANCE FROM BOTTUM/TOP//////////////////////////////////////////////////////////////////////////
//define source and sink as pixels with cerain distance from the vessel bottum and to
int MaxSinkY=MaxY-int((MaxY-MinY)*Margin);//  sink are all pixels that have Y larger then  this 
int MinSourceY=MinY+int((MaxY-MinY)*Margin);//source are all pixels that have Y smaller then  this 
/////////////////////TOP AND BOTTUM MARGINS BY AREA FRACTION//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//optional as alternative to taking the top and source as sink as pixel with minimal distance to top or bottum repectively we can take the source and sink as the bottum/top X% pixels of the vessel in term of area 
	if (UseAreaSinkSource)
	{
		 MaxSinkY=MinSourceY=0;//  redefine the boundary of the source and sink;
          vector<double> LineArea(Sides.size());// LineArea(f) is the area inside the vessel region in pixels until line Sides(f).y
          double SumArea=0;
         for (int f=0;f<Sides.size();f++) 
                {
	                        if (f>0) LineArea[f]=LineArea[f-1]+Sides[f].x2-Sides[f].x1;
	                        else LineArea[f]=Sides[f].x2-Sides[f].x1;
	                        SumArea+=Sides[f].x2-Sides[f].x1;//total area of the vessel in the image in term of pixels
                }
       for (int f=0;f<Sides.size();f++)
                 {
	                 LineArea[f]/=SumArea;
					 if  (MaxSinkY==0 &&  LineArea[f]>=1.0-SFract)    MaxSinkY=Sides[f].y;
					 if  (MinSourceY==0 &&  LineArea[f]>SFract)  MinSourceY=Sides[f].y;
                 }
	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////Build Graph////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////Calculate and add strength of nodes(pixels) to neighbours nodes(pixels) as the intensity difference between this pixels and add this to graph
Mat SinkSource=Img.clone();//for display porpuse

for (int f=1;f<Sides.size()-1;f++) 	//go over all pixel in the vesseel region of the image
	for (int x=Sides[f].x1+1;x<Sides[f].x2-1;x++)
	{
		int y=Sides[f].y;
		double Wdth=abs(int(Sides[f].x2-Sides[f].x1));// Width in pixels of current row
		int indx=MInd.at<int>(y,x);// Node index of current pixel (for the graph)
		int Dxindx=MInd.at<int>(y,x+1);//Node index of pixel to the left for the graph
		int Dyindx=MInd.at<int>(y+1,x);//Node Index of pixel below current pixel
		double Ix,Iy;
		if (Sigma>0)
		{
					Ix=exp(-SQR((Img.at<uchar>(y,x+1)-Img.at<uchar>(y,x))/Sigma)/2)*1000*KHorrizontalWeight/Wdth;//Capicity Strengh of the connection between the pixel and pixel to the left is the intesnity of difference between pixel and pixel to the left
	                Iy=exp(-SQR((Img.at<uchar>(y+1,x)-Img.at<uchar>(y,x))/Sigma)/2)*1000/Wdth;//Capicity/Strengh of the connection between the pixel and pixel to the left is  intesnity of difference between pixel and pixel above
		}
		else//Use linear weight with respect to intensity difference
		{
		Ix=(abs(255-abs(Img.at<uchar>(y,x+1)-Img.at<uchar>(y,x)))*100/Wdth);//Capicity Strengh of the connection between the pixel and pixel to the left is the intesnity of difference between pixel and pixel to the left
	    Iy=(abs(255-abs(Img.at<uchar>(y+1,x)-Img.at<uchar>(y,x)))*100/Wdth);//Capicity/Strengh of the connection between the pixel and pixel to the left is  intesnity of difference between pixel and pixel above
		   if (Sigma<0) //Optional normalize the node capicities/strength of the connection in the graph by local inensity
		  {
			Ix/=MAX(Img.at<uchar>(y,x+1),Img.at<uchar>(y,x));
			Iy/=MAX(Img.at<uchar>(y+1,x),Img.at<uchar>(y,x));
		   }
		}
		if (PenaltyZoneOn && PenaltyZone.at<uchar>(y,x)>0) Ix*=Penalty;//	Iy*=10;} //Apply penalty zone cuts near the boundary are discouraged by increasing connections capicities in this region;	
		if (indx>-1)
		{//add connection strange to the graph    
			if (Dxindx>-1) g -> add_edge( Dxindx, indx,    /* capacities */  Ix,Ix);//first and second numbers are the nodes that make the connection and third and forth numbers are the power of the connection between them (assuming symmetric connections the third and forth number should be identical)
			if (Dyindx>-1) g -> add_edge( Dyindx, indx,    /* capacities */  Iy, Iy);//first and second numbers are the nodes that make the connection and third and forth numbers are the power of the connection between them (assuming symmetric connections the third and forth number should be identical)
			if (y<MinSourceY || y>MaxSinkY) 
			{
				g -> add_tweights( indx,   /* capacities */  10000*(y<MinSourceY), 10000*(y>MaxSinkY) );//Connection to source and sink first number is node number Second number is capcity(power of connection) to sink  and third is power of connection to source
				SinkSource.at<uchar>(y,x)=255;//for display porpuse
				//-----------------------------build histogram of source and sink-------------------------------------------------------------
				 
				 if (y<=MinSourceY) {SourceHistogram[Img.at<uchar>(y,x)]++;nSource++;}//Histogram of source pixels (top Margin% pixels of the vessel)
				 if (y>=MaxSinkY) {SinkHistogram[Img.at<uchar>(y,x)]++;nSink++;}// Histogram of sink pixels (bottum Margin% pixels of the vessel)
		    }		
	}
	}
	//imshow("Ib",Ib);
	//waitKey();
	if (ShowMode) imshow("SinkSource",SinkSource);
/////////////////////Create pixels direct connection to the source and sink. Optional usefull for strongly colored materials but useless for transparent materials//////////////////////////////////////////////////////////////////////
if (k!=0) //use direct connection between source and sink
{
///calculatte connection between each pixel and the source and sink pixels and add to the graph////////////////////////////////////	
if (MixtureOfGaussian)// Caclulate connection between each pixel to source and sink using mixture of gaussian model
{
	double SigmaSqrSink=0, SigmaSqrSource=0, MeanSink=0,MeanSource=0;//Paramters for source and sink intensity gaussians
		for(int ff=0;ff<256;ff++)// Calculate mean for source and sink intensity gaussian
			{
				 MeanSink+=SinkHistogram[ff]*ff;
				MeanSource+=SourceHistogram[ff]*ff;
			}
      MeanSink/=nSink;
	  MeanSource/=nSource;
	  	for(int ff=0;ff<256;ff++)// Calculate Sigma sqr for source and sink intensity gaussian
			{
				 SigmaSqrSink+=SinkHistogram[ff]*SQR(ff-MeanSink);
				  SigmaSqrSource+=SourceHistogram[ff]*SQR(ff-MeanSource);	
			}
		  SigmaSqrSink/=nSink;
	      SigmaSqrSource/=nSource;
//...................................................................................................................
for (int f=1;f<Sides.size();f++) 	
	for (int x=Sides[f].x1+1;x<Sides[f].x2;x++)//go over all pixels in the vessel region of the image and connect them to sink or source
	{
	double Wdth=abs(int(Sides[f].x2-Sides[f].x1));// Width in pixels of current row
	int y=Sides[f].y;
		uchar In=Img.at<uchar>(y,x);
		int indx=MInd.at<int>(y,x);
		if (y>MinSourceY && y<MaxSinkY && indx>-1) 			
	    {
		double DifSink=SQR(In-MeanSink)/SigmaSqrSink;
		double DifSource=SQR(In-MeanSource)/SigmaSqrSource;
		g -> add_tweights( indx,   /* sink */int((DifSink>DifSource)*k*1000/Wdth),/* source */int((DifSink<DifSource)*k*1000/Wdth) );//Connection to source or sink first number is node number Second number is capcity  of connection to source  and third is capicity of connection to sink
		}
	}
} 
//----------------------------------------------------------------------------------------------------------------------------------------
else// Calculate every pixel connection  to source and sink according to the difference between the pixel intensity and the rest of the pixels in the source and sink
{
for (int f=1;f<Sides.size();f++) 	
	for (int x=Sides[f].x1+1;x<Sides[f].x2;x++)//go over all pixels in the vessel region of the image
	{double Wdth=abs(int(Sides[f].x2-Sides[f].x1));// Width in pixels of current row
		double Csink=0;// the total stength of the connection between corrent pixel and sink
        double Csource=0;// the total stength of the connection between corrent pixel and source
		int y=Sides[f].y;
		uchar In=Img.at<uchar>(y,x);
		int indx=MInd.at<int>(y,x);
		if (y>MinSourceY && y<MaxSinkY && indx>-1) 			
	    {
			for(int ff=0;ff<256;ff++)// calculate total connection strength of each pixel to sink and source according to intensity difference of the pixel and the sink/source histograms
			{
				Csink+=exp(-SQR((In-ff)/Sigma)/2)*1000*SinkHistogram[ff];
			  Csource+=exp(-SQR((In-ff)/Sigma)/2)*1000*SourceHistogram[ff];
			}
			Csink/=nSink;
			Csource/=nSource;
		}
		
		g -> add_tweights( indx,   /* capacities */int(Csource*k/Wdth), int(Csink*k/Wdth) );//Connection to source or sink first number is node number Second number is capcity  of connection to source  and third is capicity of connection to sink
	}
}
}
///////////////////////////////////////Run Graph Cut////////////////////////////////////////////////////////////////////////////////////////
		int flow = g -> maxflow();// calculate the graph cut for each node set value of either source or sink
//////////////////////////////////////Remove unconnecteted regions to make single sink region and one source region (very uneffecient section rewrite)/////////////////////////////////////////////////////////////

		//------------------------------remove blob of source inside sink---------------------------------------------------------------------
		Mat SourceImage;
SourceImage=Mat::zeros(Img.size(),CV_8U);// all pixels corresponding to source will be 255 in this image the rest will be zero
int SourceNum=0;// num of pixels in source for current loop
int SourceNumOld=1;
while (SourceNumOld!=SourceNum)
{
	SourceNumOld=SourceNum;
for (int f=1;f<Sides.size()-1;f++) 	
	for (int x=Sides[f].x1+1;x<Sides[f].x2;x++)
	{ 
			int y=Sides[f].y;
			if (y<MinSourceY) SourceImage.at<uchar>(y,x)=255;
	else
			{
		       int indx=MInd.at<int>(y,x);
			   	if (indx!=-1 &&  SourceImage.at<uchar>(y,x)==0 && g->what_segment(indx) == GraphType::SOURCE) // check that the pixel is connected to the source region
				{
					bool b=0;
		           for (int ty=-1;ty<=1;ty++)
				   for (int tx=-1;tx<=1;tx++)
				   {
					
					   if (SourceImage.at<uchar>(y+ty,x+tx)==255) b=true;
				   }
				   if (b)
				   {
					   SourceImage.at<uchar>(y,x)=255;
					   SourceNum++;
				   }
				}	     
			}
	}
}
//----------------------------------------------------Remove blobes of sink inside source------------------------------------------------------------------------------------------
Mat SinkImage;
SinkImage=Mat::zeros(Img.size(),CV_8U);// all pixels corresponding to source will be 255 in this image the rest will be zero
int SinkNum=0;// num of pixels in source for current loop
int SinkNumOld=1;
while (SinkNumOld!=SinkNum)
{
	SinkNumOld=SinkNum;
for (int f=Sides.size()-1;f>0;f--) 	
	for (int x=Sides[f].x1+1;x<Sides[f].x2;x++)
	{ 
			int y=Sides[f].y;
			if (y>MaxSinkY) SinkImage.at<uchar>(y,x)=255;
	else
			{
		          int indx=MInd.at<int>(y,x);
			   	if (indx!=-1 &&  SinkImage.at<uchar>(y,x)==0 && SourceImage.at<uchar>(y,x)==0) // check that the pixel is connected to the source region
				{
					bool b=0;
		           for (int ty=-1;ty<=1;ty++)
				   for (int tx=-1;tx<=1;tx++)
				   {
					
					   if (SinkImage.at<uchar>(y+ty,x+tx)==255) b=true;
				   }
				   if (b)
				   {
					   SinkImage.at<uchar>(y,x)=255;
					   SinkNum++;
				   }
				}
			}
	}
}
////////////////////////////////////Display Graph Result//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cv::cvtColor(Img, Img ,CV_GRAY2RGB);
for (int f=1;f<Sides.size()-1;f++) 	
	for (int x=Sides[f].x1+1;x<Sides[f].x2-1;x++)
	{ 
			int y=Sides[f].y;

	//		if (indx!=-1 && g->what_segment(indx) == GraphType::SOURCE) Img.at<uchar>(y,x)=100;
			    if ((SinkImage.at<uchar>(y,x) !=SinkImage.at<uchar>(y+1,x) &&  MInd.at<int>(y+1,x)!=-1)
				||  (SinkImage.at<uchar>(y,x) !=SinkImage.at<uchar>(y,x+1)  &&  MInd.at<int>(y,x+1)!=-1))
			    { Img.at<cv::Vec3b>(y,x+1)=cv::Vec3b(0,0,255);
			      Img.at<cv::Vec3b>(y,x)=cv::Vec3b(0,0,255);
			      Img.at<cv::Vec3b>(y+1,x)=cv::Vec3b(0,0,255);}// check for  node 0 if its a source or sink
//		if (SinkImage.at<uchar>(y,x)==255 ) Img.at<uchar>(y,x)=255;// check for  node 0 if its a source or sink


	}
	if (ShowMode)	{cv::imshow("Result",Img); cv::waitKey();}

	ImOut=Img;

	delete g;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//********************************************************************END OF Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************
//********************************************************************END OF Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************
//********************************************************************END OF Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************
//********************************************************************END OF Find_Phase_Boundary_By_Graph_Cut**********************************************************************************************
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphCutPhaseBoundaryOnDir(char* InpDirName, double Sigma=1,double k=0,  char * Itag=".jpg",char * Btag="_BORDERS.tif")
{
	//Run graph cut material boundary segmentation(previous function) on all the images in the directory  InpDirName and write the output image directory
	// InpDirName should contain list of  images of vessels containing materials, for each image there should be a binary image in the same directory with the same name 
	//Itag is some identifcation in the name of all files that contain the images of vessels other files only files ending with this string will be considered
	//Btag string that tag files that contain the boundary of the vessel in the image  if file named XItag is found (where X is some kind of string) looj for file named XBtag that contain the boundary of the vessel in first file
	// k is the k paramter in the graph cut function
	   DIR *dir;//directory of imahe
	   int Fcount=0;
struct dirent *ent;
cout<<"\n Proccessing Directory:"<<InpDirName<<"\n Sigma="<<Sigma<<"       k="<<k<<"\n";
if ((dir = opendir (InpDirName)) != NULL)
{
  /* find all the files and within directory */
  while ((ent = readdir (dir)) != NULL) //scan all files in library and search for files  that have the string Itag in their name, this files are used as images for the graph cut liquid boundary segmentation 
  {
	  string filename(ent->d_name);

	  std::size_t found = filename.find(Itag);// if the name tag of image to be used appear in the file name use this image otherwise ignore it
      if (found!=std::string::npos)
      {
          std::cout <<++Fcount<< ") Proccessing " << filename << '\n';
		  string ImgFileName=InpDirName+string("\\")+filename;
		  Mat Img=imread(ImgFileName);//The actual image of the transperent vessel with the material
		  
		  if (!Img.empty())
		{
		//	imshow("Img",Img); 
			ImgFileName.erase(ImgFileName.end()-strlen(Itag),ImgFileName.end());
			ImgFileName+= Btag;
			  Mat Ib=imread(ImgFileName);// the boundary of the vessel in img same as the vessel image file name only end with Btag string this files should be a binary image with pixels corresponding the boundary/contour of the vessel in previous image marked one and the rest of the pixels marked zero (should be a close contour line with line thikness of one pixel 
	     	if (!Ib.empty())
	    	{
				Mat ImgClean;//save clean unmarked version of the vessel for displaying
				resize(Img,ImgClean,Ib.size());
                Mat ImOut;
		 Find_Phase_Boundary_By_Graph_Cut(Img, Ib,ImOut,Sigma , k);

		 //.....................Write Output............................................................
			   ImgFileName.erase(ImgFileName.end()-strlen(Btag),ImgFileName.end());
			   ImgFileName+= "Output_Boundary_Marked.png";
			   imwrite(ImgFileName,ImOut);// Write the image with the boundary of the material marked red
			   ImgFileName.erase(ImgFileName.end()-strlen("Output_Boundary_Marked.png"),ImgFileName.end());
			 
			    ImgFileName+= "Resized.png";//The unmarked version of the image with the vessel contour
				   imwrite(ImgFileName,ImgClean);// Write the image with the boundary of the material marked red


		    } else cout<<"\n"<<"cant open"<<ImgFileName<<"\n";

		} else cout<<"\n"<<"cant open"<<ImgFileName<<"\n";

      }
  }
  closedir (dir);
 // int ll;
 // cin>>ll;
 } 
else {
  /* could not open directory */
  perror ("cant open directory");

}
if (strcmp(Itag,".jpg")==0) GraphCutPhaseBoundaryOnDir(InpDirName, Sigma,k,  ".JPG",Btag);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 
void main()
{



	/*Mat Img=imread("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K02\\DIMG_660d.JPG");
	Mat Ib=imread("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K02\\DIMG_660d_BORDERS.tif");
	Mat ImOut;
	
	Find_Phase_Boundary_By_Graph_Cut(Img, Ib,ImOut);
	imshow("fdfD",ImOut);
	waitKey();*/
// GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S10\\",10);
 //GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K02\\",20);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\Images for upload\\SolidsS20",20);
/* GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S30\\",30);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S40\\",40);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S50\\",50);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S60\\",60);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S70\\",70);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S80\\",80);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S90\\",90);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Liquids\\S100\\",100);*/


 /*GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S10\\",10);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S20\\",20);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S30\\",30);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S40\\",40);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S50\\",50);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S60\\",60);
  GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S70\\",70);
  GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S80\\",80);
  GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S90\\",90);
//GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeight\\Solids\\S100\\",100);*/
/*
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K00\\",20,0.0);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K02\\",20,0.02);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K04\\",20,0.04);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K07\\",20,0.07);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K10\\",20,0.1);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K15\\",20,0.15);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K20\\",20,0.2);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K30\\",20,0.3);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K40\\",20,0.4);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K60\\",20,0.6);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K100\\",20,1);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K200\\",20,2);
 GraphCutPhaseBoundaryOnDir("C:\\Users\\mithycow\\Documents\\GraphCut\\ExponentWeightBulk\\Solids\\K300\\",20,3);*/
}
