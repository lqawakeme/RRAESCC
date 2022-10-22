import pydicom
import cv2 as plt
import numpy
import os




def readcontent(rt, ct, time):
    global dcmfile
    ds = pydicom.read_file(rt)
    dsROI = ds.ROIContourSequence[3]
    for z in range(len(dsROI.ContourSequence)):
        seq = dsROI.ContourSequence[z].ContourImageSequence[0].ReferencedSOPInstanceUID
        for root10, dirs10, files10 in os.walk(ct):
                for file3 in files10:
                    if str(file3).find((seq +'.dcm'), 0, len(str(file3))) > -1:
                        dcmfile = str(os.path.join(root10, file3))
        fileisExists = os.path.exists(dcmfile)
        if fileisExists:
            ds2 = pydicom.read_file(dcmfile)
            dcmOrigin = ds2.ImagePositionPatient
            dcmSpacing = ds2.PixelSpacing
            content = ds.ROIContourSequence[3].ContourSequence[z]
            points = content.NumberOfContourPoints
            world = numpy.zeros((points, 3), int)
            twod = numpy.zeros((points, 2), int)
            for k in range(points):
                m = (k - 1) * 3
                world[k, 0] = content.ContourData[m]
                world[k, 1] = content.ContourData[m + 1]
                world[k, 2] = content.ContourData[m + 2]
                twod[k, 0] = round((world[k, 0] - dcmOrigin[0]) / dcmSpacing[0])
                twod[k, 1] = round((world[k, 1] - dcmOrigin[1]) / dcmSpacing[1])
            img = ds2.pixel_array
            img[img == -2000] = 0
            org_img = img
            path = '/Users/liuqiang/Desktop/origin_pngs/' +str(ds.PatientID)+'_'+ str(time)+"_" + str(ds.PatientName) + '_' + str(unix)
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            plt.imwrite(('/Users/liuqiang/Desktop/origin_pngs/' +str(ds.PatientID)+'_'+ str(time)+"_" + str(ds.PatientName) + '_' + str(unix)+ '/image_' + str(z) + '.png'),
                        org_img)
            img3 = plt.imread('/Users/liuqiang/Desktop/origin_pngs/' +str(ds.PatientID)+'_'+ str(time)+"_" + str(ds.PatientName) + '_' + str(unix)+ '/image_' + str(z) + '.png')
            img2 = numpy.zeros((512, 512), numpy.uint8)
            img2 = plt.cvtColor(img2, plt.COLOR_GRAY2BGR)
            img2[:, :, :] = 255
            img4 = img3.copy()
            plt1 = []
            row_max = twod[0,0]
            row_min = twod[0, 0]
            column_max = twod[0, 1]
            column_min = twod[0, 1]
            for l in range(points):
                plt1.append((twod[l, 0], twod[l, 1]))
                row_max = max(twod[l, 0], row_max)
                row_min = min(twod[l, 0], row_min)
                column_max = max(twod[l, 1], column_max)
                column_min = min(twod[l, 1], column_min)
            for l in range(points-1):
                plt.line(img4, (twod[l, 0], twod[l, 1]), (twod[l + 1, 0], twod[l + 1, 1]), (0, 255, 0), 1, 4)

            nppoints = numpy.array(plt1, numpy.int32)
            mask = numpy.zeros(img3.shape,numpy.uint8)
            mask2 = plt.fillPoly(mask.copy(), [nppoints], (255, 255, 255))
        #    mask3 = plt.fillPoly(mask.copy(), [nppoints], (0, 255, 0))
            ROI = plt.bitwise_and(mask2, img3)
            if (column_max > column_min) & (row_max > row_min) :
                cut_ROI = ROI[column_min:column_max, row_min:row_max]
                path2 = '/Users/liuqiang/Desktop/final_pngs/' + str(ds.PatientID) + '_' + str(time) + "_" + str(
                    ds.PatientName) + '_' + str(unix)
                isExists2 = os.path.exists(path2)
                if not isExists2:
                    os.makedirs(path2)
                plt.imwrite(('/Users/liuqiang/Desktop/final_pngs/' + str(ds.PatientID) + '_' + str(time) + "_" + str(
                    ds.PatientName) + '_' + str(unix) + '/image_' + str(z) + '.png'), cut_ROI)
            else :
                print(str(ds.PatientID) + '_' + str(time) + "_" + str(
                    ds.PatientName) + '_' + str(z))
                print(str(row_min), str(row_max), str(column_min), str(column_max))
            path2 = '/Users/liuqiang/Desktop/masked_pngs/'+str(ds.PatientID)+'_'+ str(time)+"_" + str(ds.PatientName)+'_'+str(unix)
            isExists2 = os.path.exists(path2)
            if not isExists2:
                os.makedirs(path2)
            plt.imwrite(('/Users/liuqiang/Desktop/masked_pngs/'+str(ds.PatientID)+'_'+ str(time)+"_" + str(ds.PatientName)+'_'+str(unix) + '/image_' + str(z) + '.png'), ROI)
            path3 = '/Users/liuqiang/Desktop/mask/'+str(ds.PatientID)+'_'+ str(time)+"_" + str(ds.PatientName) + '_' + str(unix)
            isExists2 = os.path.exists(path3)
            if not isExists2:
                os.makedirs(path3)
            plt.imwrite(
            ('/Users/liuqiang/Desktop/mask/'+str(ds.PatientID)+'_'+ str(time)+"_" + str(ds.PatientName) + '_' + str(unix) + '/image_' + str(z) + '.png'),
            mask2)
            path4 = '/Users/liuqiang/Desktop/lined_pngs/'+ str(ds.PatientID)+'_'+str(time)+"_" + str(ds.PatientName) + '_' + str(unix)
            isExists2 = os.path.exists(path4)
            if not isExists2:
                os.makedirs(path4)
            plt.imwrite(
                ('/Users/liuqiang/Desktop/lined_pngs/'+str(ds.PatientID)+'_'+ str(time)+"_" + str(ds.PatientName) + '_' + str(unix) + '/image_' + str(
                    z) + '.png'),
                img4)

        else:
            print(seq+"        not exist.")


temp_string = []
temp_string2 = []
temp_dir = []
temp_root = []
time_list = []
m=0
for root, dirs, files in os.walk("/Users/liuqiang/Desktop/151-180"):
    for dir in dirs:
        if str(dir).find('_RTst_', 0, len(str(dir))) > 0:
            tem_time = str(dir).split("_")
            time = tem_time[3]
           # print(time)
            time_list.append(time)
            temp_root.append(root)
            for root2, dirs2, files2 in os.walk(os.path.join(root, dir)):
                for file in files2:
                    if str(file).find('dcm', 0, len(str(file))) > 0:
                        RTstr = str(os.path.join(os.path.join(root, dir), file))
                        temp_string.append(RTstr)
                        temp_dir.append(str(dir))
                        m=m+1
#print(m)
unix = 0
for ll in range(m):
    for root, dirs, files in os.walk("/Users/liuqiang/Desktop/151-180"):
        for dir2 in dirs:
            num = str(temp_dir[ll]).find('_',0,len(temp_dir[ll]))
            num2 =str(temp_dir[ll]).find('_', num+1 ,len(temp_dir[ll]))
            num3 =str(temp_dir[ll]).find('_', num2+1 ,len(temp_dir[ll]))
            num4 = str(temp_dir[ll]).find('_', num3+1, len(temp_dir[ll]))
            if str(dir2).find(str(temp_dir[ll])[0:num2], 0, len(str(dir2))) > -1 :
                if str(dir2).find(str(temp_dir[ll])[num3:num4], 0, len(str(dir2))) > -1:
                    if str(dir2).find('CT', 0,len(str(dir2))) > -1:
                        temp_string2.append(str(os.path.join(root, dir2)))
print(len(temp_string))
print(len(temp_root))
for zz in range(m):
 #   print(temp_string[zz], temp_root[zz])
    readcontent(temp_string[zz], temp_root[zz], time_list[zz])
    unix = unix + 1
print(1)
