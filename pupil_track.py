import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('Vertical Nystagmus.mp4')

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# size = (frame_width, frame_height)

# result = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)

def fit_rotated_ellipse_ransac(data,iter=50,sample_num=10,offset=80.0):

    count_max = 0
    effective_sample = None

    for i in range(iter):
        sample = np.random.choice(len(data), sample_num, replace=False)

        xs = data[sample][:,0].reshape(-1,1)
        ys = data[sample][:,1].reshape(-1,1)

        J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
        Y = np.mat(-1*xs**2)
        P= (J.T * J).I * J.T * Y

        # fitter a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
        a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
        ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

        # threshold 
        ran_sample = np.array([[x,y] for (x,y) in data if np.abs(ellipse_model(x,y)) < offset ])

        if(len(ran_sample) > count_max):
            count_max = len(ran_sample) 
            effective_sample = ran_sample

    return fit_rotated_ellipse(effective_sample)


def fit_rotated_ellipse(data):

    xs = data[:,0].reshape(-1,1) 
    ys = data[:,1].reshape(-1,1)

    J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
    Y = np.mat(-1*xs**2)
    P= (J.T * J).I * J.T * Y

    a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
    theta = 0.5* np.arctan(b/(a-c))  
    
    cx = (2*c*d - b*e)/(b**2-4*a*c)
    cy = (2*a*e - b*d)/(b**2-4*a*c)

    cu = a*cx**2 + b*cx*cy + c*cy**2 -f
    w= np.sqrt(cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2))
    h= np.sqrt(cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2))

    ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

    error_sum = np.sum([ellipse_model(x,y) for x,y in data])
    print('fitting error = %.3f' % (error_sum))

    return (cx,cy,w,h,theta)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
xcoordinates= []
ycoordinates= []
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
      ret, frame = cap.read()
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
      image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(image_gray,(3,3),0)
      ret,thresh1 = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
      opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
      closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

      image = 255 - closing
      _,contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      hull = []

      for i in range(len(contours)):
          hull.append(cv2.convexHull(contours[i], False)) 
                      
    #   cnt = sorted(hull, key=cv2.contourArea)
    #   maxcnt = cnt[-1]
      for con in hull:
          approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con,True),True)
          area = cv2.contourArea(con)
          if(len(approx) > 10 and area > 1000):
              cx,cy,w,h,theta = fit_rotated_ellipse_ransac(con.reshape(-1,2))
              xcoordinates.append(cx)
              ycoordinates.append(cy)
              cv2.ellipse(frame,(int(cx),int(cy)),(int(w),int(h)),theta*180.0/np.pi,0.0,360.0,(0,0,255),1)
              cv2.drawMarker(frame, (int(cx),int(cy)),(0, 0, 255),cv2.MARKER_CROSS,2,1)
              cv2.imshow('Output',frame)
              
              
            #   result.write(frame)
    # Press Q on keyboard to  exit
      
      if cv2.waitKey(25) & 0xFF == ord('q'):
           break   
  # Break the loop
  else: 
    break

plt.plot(xcoordinates[:])
plt.xlabel
plt.show()
plt.plot(ycoordinates[:])

# When everything done, release the video capture object
cap.release()
# result.release()
# Closes all the frames
cv2.destroyAllWindows()
