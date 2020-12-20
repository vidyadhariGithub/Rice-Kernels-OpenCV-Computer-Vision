# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author: 
"""

#......IMPORT .........
import argparse
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import statistics 
import operator
import os
import glob


#Iso-data Intensity Thresholding
def task1(img1, output_folder, filename):
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    meanval = cv2.mean(gray)
    initial_threshold = meanval[0]
    r_t = initial_threshold

    #calculate mean of white and black pixels
    new_threshold = 1
    iterations = []
    thresholds = []
    iterations.append(0)
    thresholds.append(initial_threshold)
    count = 0
    new_threshold = initial_threshold
    converged = False

    while(converged != True):
        img = gray.copy()
        r_t = new_threshold
        white_sum = black_sum = 0
        white_count = black_count = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                val = img[i,j]
                if val < r_t: 
                    white_sum = white_sum + val
                    white_count = white_count + 1
                else:
                    black_sum = black_sum + val
                    black_count = black_count + 1
        white_mean = 0
        black_mean = 0
        if white_count != 0:
            white_mean = white_sum/white_count
        if black_count != 0:
            black_mean = black_sum/black_count
        previous_threshold = new_threshold
        new_threshold = (white_mean + black_mean) /2
        count = count + 1
        iterations.append(count)
        thresholds.append(new_threshold)
        x = abs(new_threshold - previous_threshold)
        if  x <= 0.01:
            converged = True

    r_t = new_threshold
    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            r = gray[x,y]
            if r < r_t: 
                gray[x,y] = 255 
            elif r >= r_t: 
                gray[x,y] = 0 

    file = filename[:-4] + '_Task1.png'
    val = round(r_t,1)
    text = 'Threshold Value ='+ str(val)

    plt.imshow(gray,'gray')
    plt.title(text)
    plt.xticks([]),plt.yticks([])
    plt.savefig(output_folder+'/'+file)
    plt.show()
    
    return gray, iterations, thresholds
    
    
def Median_Filter(data,size):    
    x = data.shape[0]
    y = data.shape[1]
    f = size//2
    i = 0
    kernel = []
    while i < x:
        j = 0
        while j < y:
            for P1 in range(i - f, i + f + 1):
                for P2 in range(j - f, j + f + 1):
                    if ((P1 >= 0 and P1 < x) and (P2>=0 and P2< y)):
                        kernel.append(data[P1][P2])
                    else:
                        kernel.append(0)
            kernel.sort()
            pos = len(kernel)//2
            data[i][j] = kernel[pos]
            kernel = []
            j = j + 1
        i = i + 1
    return data


def task2(gray, output_folder, filename):
    filtered_image = Median_Filter(gray, 3)
    #filtered_image = cv2.medianBlur(gray, 5)
    cls = 1 
    for x in range(0,filtered_image.shape[0]):
        for y in range(0,filtered_image.shape[1]):
            r = filtered_image[x][y]
            l = []
            #pixel is not the background, get its non-background neighbours
            if r == 0: #rice grain found 
                if (x-1 >0 and x-1 < filtered_image.shape[0]-1) and (y-1 > 0 and y-1 < filtered_image.shape[1]-1):
                    if filtered_image[x-1][y-1] >= 0 and filtered_image[x-1][y-1] != 255:
                        l.append(filtered_image[x-1][y-1])
                if (x-1 >0 and x-1 < filtered_image.shape[0]-1) and (y > 0 and y < filtered_image.shape[1]-1):
                    if filtered_image[x-1][y] >= 0 and filtered_image[x-1][y] != 255:
                        l.append(filtered_image[x-1][y])
                if (x-1 >0 and x-1 < filtered_image.shape[0]-1) and (y+1 > 0 and y+1 < filtered_image.shape[1]-1):
                    if filtered_image[x-1][y+1] >= 0 and filtered_image[x-1][y+1] != 255:
                        l.append(filtered_image[x-1][y+1])
                if (x >0 and x < filtered_image.shape[0]-1) and (y-1 > 0 and y-1 < filtered_image.shape[1]-1):
                    if filtered_image[x][y-1] >=0 and filtered_image[x][y-1] != 255:
                        l.append(filtered_image[x][y-1])
                if (x >0 and x < filtered_image.shape[0]-1) and (y+1 > 0 and y+1 < filtered_image.shape[1]-1):
                    if filtered_image[x][y+1] >=0 and filtered_image[x][y+1] != 255:
                        l.append(filtered_image[x][y+1])
                if (x+1 >0 and x+1 < filtered_image.shape[0]-1) and (y-1 > 0 and y-1 < filtered_image.shape[1]-1):
                    if filtered_image[x+1][y-1] >=0 and filtered_image[x+1][y-1] != 255:
                        l.append(filtered_image[x+1][y-1])
                if (x+1 >0 and x+1 < filtered_image.shape[0]-1) and (y > 0 and y < filtered_image.shape[1]-1):
                    if filtered_image[x+1][y] >= 0 and filtered_image[x+1][y] != 255:
                        l.append(filtered_image[x+1][y])
                if (x+1 >0 and x+1 < filtered_image.shape[0]-1) and (y+1 > 0 and y+1 < filtered_image.shape[1]-1):
                    if filtered_image[x+1][y+1] >= 0 and filtered_image[x+1][y+1] != 255:
                        l.append(filtered_image[x+1][y+1])
                if len(l) != 0: #neighbors is empty
                    if max(l) == 0: #all black neighbors
                        filtered_image[x][y] = cls
                        cls = cls+1
                        if cls == 255:
                            cls = 256
                    else:
                        sorted_l = sorted(list(set(l)))
                        if sorted_l[0] == 0:
                            filtered_image[x][y] = sorted_l[1]
                        else:
                            filtered_image[x][y] = sorted_l[0]

    #Second Pass  
    converged = False
    while converged == False:
        converged = True
        for x in range(0,filtered_image.shape[0]):
            for y in range(0,filtered_image.shape[1]):
                l = []
                if filtered_image[x][y] != 255:
                    if (x-1 >0 and x-1 < filtered_image.shape[0]-1) and (y-1 > 0 and y-1 < filtered_image.shape[1]-1):
                        if filtered_image[x-1][y-1] >= 0 and filtered_image[x-1][y-1] != 255:
                            l.append(filtered_image[x-1][y-1])
                    if (x-1 >0 and x-1 < filtered_image.shape[0]-1) and (y > 0 and y < filtered_image.shape[1]-1):
                        if filtered_image[x-1][y] >= 0 and filtered_image[x-1][y] != 255:
                            l.append(filtered_image[x-1][y])
                    if (x-1 >0 and x-1 < filtered_image.shape[0]-1) and (y+1 > 0 and y+1 < filtered_image.shape[1]-1):
                        if filtered_image[x-1][y+1] >= 0 and filtered_image[x-1][y+1] != 255:
                            l.append(filtered_image[x-1][y+1])
                    if (x >0 and x < filtered_image.shape[0]-1) and (y-1 > 0 and y-1 < filtered_image.shape[1]-1):
                        if filtered_image[x][y-1] >=0 and filtered_image[x][y-1] != 255:
                            l.append(filtered_image[x][y-1])
                    if (x >0 and x < filtered_image.shape[0]-1) and (y+1 > 0 and y+1 < filtered_image.shape[1]-1):
                        if filtered_image[x][y+1] >=0 and filtered_image[x][y+1] != 255:
                            l.append(filtered_image[x][y+1])
                    if (x+1 >0 and x+1 < filtered_image.shape[0]-1) and (y-1 > 0 and y-1 < filtered_image.shape[1]-1):
                        if filtered_image[x+1][y-1] >=0 and filtered_image[x+1][y-1] != 255:
                            l.append(filtered_image[x+1][y-1])
                    if (x+1 >0 and x+1 < filtered_image.shape[0]-1) and (y > 0 and y < filtered_image.shape[1]-1):
                        if filtered_image[x+1][y] >= 0 and filtered_image[x+1][y] != 255:
                            l.append(filtered_image[x+1][y])
                    if (x+1 >0 and x+1 < filtered_image.shape[0]-1) and (y+1 > 0 and y+1 < filtered_image.shape[1]-1):
                        if filtered_image[x+1][y+1] >= 0 and filtered_image[x+1][y+1] != 255:
                            l.append(filtered_image[x+1][y+1])
                    if len(l) != 0: #neighbors is empty
                        if filtered_image[x][y] != min(l):
                            converged = False
                            filtered_image[x][y] = min(l)
    kernels = []
    for x in range(0,filtered_image.shape[0]):
        for y in range(0,filtered_image.shape[1]):  
            if filtered_image[x][y] != 255:
                kernels.append(filtered_image[x][y])
    
    kernel_areas = []
    unique_kernels = set(kernels)

    for k in unique_kernels:
        area = 0
        for x in range(0,filtered_image.shape[0]):
            for y in range(0,filtered_image.shape[1]): 
                if filtered_image[x][y] == k:
                    area = area+1
        kernel_areas.append(area)
    #print("kernel areas ", kernel_areas)
    i = 0
    unique_kernelss = list(unique_kernels)
    unique_kernels_final = []
    for x in kernel_areas:
        if x>25:
            unique_kernels_final.append(unique_kernelss[i])
        i = i+1
    
    #val = round(len(set(kernels)),2)
    val = len(unique_kernels_final)
    text = "Number of rice kernels = " + str(val)
    filtered_image1 = filtered_image.copy()
    k = list(set(kernels))
    for x in range(0,filtered_image1.shape[0]):
        for y in range(0,filtered_image1.shape[1]):
            if filtered_image1[x][y] in k:
                filtered_image1[x][y] = 0
    
    #filtered_image_RGB = cv2.cvtColor(filtered_image1, cv2.COLOR_BGR2GRAY)
    plt1.imshow(filtered_image1, cmap='gray')
    plt1.title(text)
    plt1.xticks([]),plt.yticks([])
    file = filename[:-4] + '_Task2.png'
    plt1.savefig(output_folder+'/'+file)
    plt1.show()
                            
    return filtered_image,kernels

    
def task3(kernels, filtered_image, filename, output_folder, min_area):
    kernel_areas = []
    unique_kernels = set(kernels)

    for k in unique_kernels:
        area = 0
        for x in range(0,filtered_image.shape[0]):
            for y in range(0,filtered_image.shape[1]): 
                if filtered_image[x][y] == k:
                    area = area+1
        kernel_areas.append(area)
    
    min_area = max(kernel_areas)//2
    damaged_kernel_count = 0
    l4 = []
    total = len(kernel_areas)
    i = 0
    l5 = list(unique_kernels)
    for x in kernel_areas:
        if x<min_area:
            damaged_kernel_count = damaged_kernel_count+1
            l4.append(l5[i])
        i=i+1
        
    damaged_percentage = (damaged_kernel_count/total)*100
    #print("Number of damaged kernels : ", damaged_kernel_count)
    val = round(damaged_percentage,2)
    text = "Percentage of damaged kernels : " + str(val)

    #binary image excluding all damaged kernels
    for x in range(0,filtered_image.shape[0]):
        for y in range(0,filtered_image.shape[1]):
            if filtered_image[x][y] in l4:
                filtered_image[x][y] = 255

    for x in range(0,filtered_image.shape[0]):
        for y in range(0,filtered_image.shape[1]):
            if filtered_image[x][y] in l5:
                filtered_image[x][y] = 0
    
    #filtered_image_RGB = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    plt2.imshow(filtered_image, cmap='gray')
    plt2.title(text)
    plt2.xticks([]),plt.yticks([])
    file = filename[:-4] + '_Task3.png'
    plt2.savefig(output_folder+'/'+file)
    plt2.show()
    
    
#calling the three tasks
my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o','--OP_folder', type=str,help='Output folder name', default = 'OUTPUT')
my_parser.add_argument('-m','--min_area', type=int,action='store', required = True, help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f','--input_filename', type=str,action='store', required = True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()

file = args.input_filename
min_area = int(args.min_area)
output_folder = args.OP_folder

if os.path.isdir('./'+output_folder) == False: 
    os.mkdir(output_folder)

img1 = cv2.imread(file, 1)

#task1
gray, iterations, thresholds = task1(img1, output_folder, file)
#display iterations vs threshold curve
plt1.xticks(iterations)
plt1.plot(iterations, thresholds)
plt1.show()

#task2
filtered_image, kernels = task2(gray, output_folder, file)

#task3
task3(kernels, filtered_image, file, output_folder, min_area)

