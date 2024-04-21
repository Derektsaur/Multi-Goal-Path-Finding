
import cv2
import PIL.Image
import os
# g = 2
# targetCoordinate = [[18,37],[54,217],[226,28],[93,111],[247,125],[192,228]]
# g = 15
# targetCoordinate = [[226,40],[237,17],[81,171],[98,64],[175,222],[157,152],[88,29],[59,236],[141,220],[9,2],[199,216],[237,168],[68,11],[119,27],[11,105],[183,6],[187,175],[130,122],[13,162],[13,221],[207,108],[147,68],[238,207],[231,120],[37,249],[94,235],[158,137]]
# g = 14
targetCoordinate = [[36, 36], [50, 168], [14, 235], [131, 165], [223, 51], [196, 235], [245, 226], [98, 94], [157, 74],
                    [235, 124]]

# g = 11
# targetCoordinate = [[102, 18], [23, 23], [100, 120], [95, 177], [145, 145], [235, 34], [249, 130], [170, 230],
#                     [14, 170], [26, 225], [240, 230], [200, 100],[81,69],[191,171]]

# g = 31   # VVV    T3851
# targetCoordinate = [[27,27],[21,236],[247,61],[174,25],[160,177],[231,175],[90,223],[69,135],[126,145]]
#
# g = 36   #VVV   W1200
# targetCoordinate = [[97,9],[56,86],[103,180],[21,25],[147,25],[96,84],[237,63],[14,224],[231,149],[245,219],[145,115],[93,245],[50,187],[185,185],[16,182],[210,246],[163,245],[215,15],[20,98],[47,140],[136,238]]
#
#
# vec = g
# route = r"C:\Users\huang\Desktop\S&Reg V2\PAPER\DRAW\BUILDING"
# # img = cv2.imread(os.path.join(route,"%d.jpg"%vec))
# img = cv2.imread(os.path.join(route,"W1200.png"))
#
# for i in range(len(targetCoordinate)):
#     if i == 0:
#         cv2.rectangle(img, (targetCoordinate[i][1] - 6, targetCoordinate[i][0] - 6),
#                       (targetCoordinate[i][1] + 6, targetCoordinate[i][0] + 6), (255, 0, 0),
#                       thickness=-1)
#     else:
#         cv2.rectangle(img,(targetCoordinate[i][1]-6,targetCoordinate[i][0]-6),(targetCoordinate[i][1]+6,targetCoordinate[i][0]+6), (0, 0, 255),
#                   thickness=-1)
#
# img = PIL.Image.fromarray(img).convert('RGB')
# img.save(os.path.join(route,"W1200_draw.jpg"), dpi=(300, 300), quality=95)


# img1 = cv2.imread(r"C:\Users\huang\Desktop\S&Reg V2\PAPER\DRAW\map_11_line.jpg")
# img1 = cv2.imread(r"C:\Users\huang\Desktop\S&Reg V2\PAPER\DRAW\map_11_line+region.jpg")
img1 = cv2.imread(r"C:\Users\huang\Desktop\S&Reg V2\PAPER\DRAW\model performance\map_11_region2.jpg")
t = [223,51]
g = [50,168]

cv2.rectangle(img1, (t[1] - 6, t[0] - 6),
              (t[1] + 6, t[0] + 6), (255, 0, 0),
              thickness=-1)
cv2.rectangle(img1, (g[1] - 6, g[0] - 6),
              (g[1] + 6, g[0] + 6), (0, 0, 255),
              thickness=-1)

img1 = PIL.Image.fromarray(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
img1.save(r'C:\Users\huang\Desktop\S&Reg V2\PAPER\DRAW\model performance\map_11_region2.jpg', dpi=(300, 300), quality=95)
