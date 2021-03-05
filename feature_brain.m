clc;
close all;
clear all;
warning('off','all');

a11 = dir('test brain\*.jpg');
 feature_final1=[];
for k = 1:length(a11)
%%TO READ THE INPUT IMAGE%%%
% % [FileName,PathName] = uigetfile('*.jpg;*.png;*.bmp','Pick an MRI Image');
% % a = imread([PathName,FileName]);
a = imread(fullfile('test brain',a11(k).name));

%%TO RESIZE THE INPUT IMAGE%%
a=imresize(a,[256 256]);
% figure,imshow(a),title('INPUT IMAGE');

%% TO CONVERT THE INPUT IMAGE INTO GRAYSCALE IMAGE%%%
[m n o]=size(a);
if o==3
    gray=rgb2gray(a);
else 
    gray=a;
end

%  figure,imshow(gray),title('GRAY SCALE IMAGE');


%%%REGION PROPERTY BASED SEGMENTATION%%%%
    %TO ADJUST THE GRAY IMAGE%%
bw1=imadjust(gray);
% figure,imshow(bw1),title('CONTRAST ENHANCED GRAYSCALE IMAGE');
    %%CONVERT TO BLACK AND WHITE IMAGE%%%
bw=im2bw(bw1,0.7);
% figure,imshow(bw),title('BLACK AND WHITE IMAGE');
    %%LABEL THE BLACK AND WHITE IMAGE%%%
label=bwlabel(bw);
% figure,imshow(label),title('LABELED IMAGE');

    %%SELECT THE REGION PROPERTY%%%%
       %%TO CALL THE ALL PROPERTIES%%
stats=regionprops(label,'all');
       %%SELECT IMPORTANT PROPERTIES%%
area=[stats.Area];
centroid=[stats.Centroid];
majorAxisLength=[stats.MajorAxisLength];
minorAxisLength=[stats.MinorAxisLength];
eccentricity=[stats.Eccentricity];
orientation=[stats.Orientation];
filledArea=[stats.FilledArea];
equivdiameter=[stats.EquivDiameter];
density=[stats.Solidity];
perimeter=[stats.Perimeter];

%%FROM THE ALL PROPERTIES WE CHOOSE 'AREA' AND 'DENSITY' %%%
  %%CHOOSE THE HIGH DENSE AREA%%
high_dense_area=density>0.3;
  %%CHOOSE THE MAX AREA WITH HIGH DENSITY%%
max_area=max(area(high_dense_area));
  %%FIND THE MAXIMUM AREA%%
tumour_label=find(area==max_area);
  %%LABEL THAT THE MAXIMUM AREA%%
tumour=ismember(label,tumour_label);
 %%DILATE  THE TUMOR AREA%%%
se=strel('square',8);
tumour=imdilate(tumour,se);
 %%TO SHOW THE TUMOR AREA ALONE%%
% figure,imshow(tumour,[]),title('TUMOR ALONE IMAGE');

 %%TO SHOW THE DETECTED TUMOR%%%%
% [B,L]=bwboundaries(tumour,'noholes');
% figure,imshow(a,[]);
% hold on
% for  i=1:length(B)
%     plot(B{i}(:,2),B{i}(:,1),'g','linewidth',1.5)
% end
% title('DETECTED TUMOUR IMAGE')
% hold off

% figure,
% subplot(1,3,1),imshow(a,[]),title('INPUT TUMOR IMAGE');
% subplot(1,3,2),imshow(tumour,[]),title('TUMOUR ALONE IMAGE');
% [B,L]=bwboundaries(tumour,'noholes');
% subplot(1,3,3),imshow(a,[]);
% hold on
% for  i=1:length(B)
%     plot(B{i}(:,2),B{i}(:,1),'g','linewidth',1.5)
% end
% title('TUMOUR DETECTED')
% hold off
% % end
% % % FEATURE EXTRACTION BY PCA AND GLCM%%%%%

     %%FEATURE EXTRACTION BY PCA(PRINCIPAL COMPONENT ANALYSIS)%%%
signal1=tumour(:,:);
[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');
DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);


     %%%FEATURE EXTRACTION BY GLCM(GRAY LEVEL CO-OCCURENCE MATRIX)%%%%
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
Skewness = skewness(G)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));

% Inverse Difference Movement
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
feat_disease1 = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];


% % Color Texture Feature Extraction(LBP) FOR TUMOR ALONE IMAGE%%%
% % step 1: Local Binary Patterns for gray image
feat_LBP = extractLBPFeatures(tumour);
grayImage = tumour;
localBinaryPatternImage1 = zeros(size(grayImage));
[row col] = size(grayImage);
for r = 2 : row - 1   
	for c = 2 : col - 1    
		centerPixel = grayImage(r, c);
		pixel7 = grayImage(r-1, c-1) > centerPixel;  
		pixel6 = grayImage(r-1, c) > centerPixel;   
		pixel5 = grayImage(r-1, c+1) > centerPixel;  
		pixel4 = grayImage(r, c+1) > centerPixel;     
		pixel3 = grayImage(r+1, c+1) > centerPixel;    
		pixel2 = grayImage(r+1, c) > centerPixel;      
		pixel1 = grayImage(r+1, c-1) > centerPixel;     
		pixel0 = grayImage(r, c-1) > centerPixel;       
		localBinaryPatternImage1(r, c) = uint8(...
			pixel7 * 2^7 + pixel6 * 2^6 + ...
			pixel5 * 2^5 + pixel4 * 2^4 + ...
			pixel3 * 2^3 + pixel2 * 2^2 + ...
			pixel1 * 2 + pixel0);
	end  
end 

% figure,imshow(localBinaryPatternImage1),title('LBP V-CHANNEL IMAGE'); 






feat_disease=[ feat_disease1 feat_LBP];

feature_final1=[feature_final1;feat_disease];
end



save 'featurebrain2.mat'  feature_final1




