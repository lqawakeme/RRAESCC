import pydicom
import cv2 as plt
import numpy
import os
import xlsxwriter as xw




def readcontent(rt, ct, time):
    global dcmfile
    ds = pydicom.read_file(rt)
    worksheet1.write(unix, 0, ds.PatientID)
    worksheet1.write(unix, 1, str(ds.PatientName))
    worksheet1.write(unix, 2, ds.PatientSex)

workbook = xw.Workbook('Patient_Sex_test.xlsx')
worksheet1 = workbook.add_worksheet("sheet1")
worksheet1.activate()
unix = 0
# temp_string = []
# temp_string2 = []
# temp_dir = []
# temp_root = []
# time_list = []
# m=0
# for root, dirs, files in os.walk("/Users/liuqiang/Desktop/151-180"):
#     for dir in dirs:
#         if str(dir).find('_RTst_', 0, len(str(dir))) > 0:
#             tem_time = str(dir).split("_")
#             time = tem_time[3]
#            # print(time)
#             time_list.append(time)
#             temp_root.append(root)
#             for root2, dirs2, files2 in os.walk(os.path.join(root, dir)):
#                 for file in files2:
#                     if str(file).find('dcm', 0, len(str(file))) > 0:
#                         RTstr = str(os.path.join(os.path.join(root, dir), file))
#                         temp_string.append(RTstr)
#                         temp_dir.append(str(dir))
#                         m=m+1
# #print(m)
#

# for ll in range(m):
#     for root, dirs, files in os.walk("/Users/liuqiang/Desktop/151-180"):
#         for dir2 in dirs:
#             num = str(temp_dir[ll]).find('_',0,len(temp_dir[ll]))
#             num2 =str(temp_dir[ll]).find('_', num+1 ,len(temp_dir[ll]))
#             num3 =str(temp_dir[ll]).find('_', num2+1 ,len(temp_dir[ll]))
#             num4 = str(temp_dir[ll]).find('_', num3+1, len(temp_dir[ll]))
#             if str(dir2).find(str(temp_dir[ll])[0:num2], 0, len(str(dir2))) > -1 :
#                 if str(dir2).find(str(temp_dir[ll])[num3:num4], 0, len(str(dir2))) > -1:
#                     if str(dir2).find('CT', 0,len(str(dir2))) > -1:
#                         temp_string2.append(str(os.path.join(root, dir2)))
# print(len(temp_string))
# print(len(temp_root))
# for zz in range(m):
#  #   print(temp_string[zz], temp_root[zz])
#     readcontent(temp_string[zz], temp_root[zz], time_list[zz])
#     unix = unix + 1

temp_string = []
temp_string2 = []
temp_dir = []
temp_root = []
time_list = []
doc_list = []
m=0
for root, dirs, files in os.walk("/Users/liuqiang/Desktop/test_group"):
    for file in files:
        if str(file).find('RS') > -1:
            temp_string.append(str(os.path.join(root,file)))
            temp_root.append(str(root))
            temp_doc_list = str(root).split('/')
            doc = temp_doc_list[len(temp_doc_list) - 1]
            doc_list.append(doc)
#            print(str(os.path.join(root,file)),str(root))
print(len(temp_string))
print(len(temp_root))
for zz in range(len(temp_string)):
  #   print(temp_string[zz], temp_root[zz])
     readcontent(temp_string[zz], temp_root[zz],doc_list[zz])
     unix = unix + 1
print('finished')
workbook.close()
print(1)
