import cv2
import numpy as np

def euler2mat(angles):
	c, s = np.cos(angles[0]), np.sin(angles[0])
	R1 = np.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
	c, s = np.cos(angles[1]), np.sin(angles[1])
	R2 = np.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
	c, s = np.cos(angles[2]), np.sin(angles[2])
	R3 = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
	return R1 * R2 * R3

def scale2mat(scale):
	return np.matrix([[scale,0,0],[0,scale,0],[0,0,1]])

def trans2mat(trans):
	return np.matrix([[0,0,trans[0]],[0,0,trans[1]],[0,0,0]])

def fill_mapping(img, mapx, mapy, color_ind=2):
    indx = np.floor(mapx).astype(int)
    indy = np.floor(mapy).astype(int)
    indx[indx >= img.shape[1]] = 0
    indy[indy >= img.shape[0]] = 0
    img[indy,indx,color_ind] = 255


class VideoCamera(object):
	def __init__(self, w, h):
		self.size_out = (608, 608)

		self.proj_wide  = 90 # 0:360
		self.proj_long  = 70 # 0:90
		self.view_angle = 45 # 0:90, 0 - top down view

		self.calibrated = True
		self.angles = [0,0,0]
		self.transl = [0,0]
		self.scale = 1.0
		self.projection = 0

	def initUndistortRectifyMap(self, rvec, P, size_out):
		return cv2.fisheye.initUndistortRectifyMap(
		    self.K,
		    self.D,
		    euler2mat(np.radians(rvec)),
		    P,
		    size_out,
		    cv2.CV_32F
		)

	def fixedRoI(self, dir, width, height):
		wide = np.radians(width / 2 + 2)
		tall = np.radians(90-height)
		dir = np.radians(dir)

		# 3d points x,y,z. We are only interested in
		# the direction to a point
		pts = np.matrix([
			[0, 0, 1.0], # center
			[np.cos(dir + wide), np.sin(dir + wide), np.tan(tall)],  # -wide
			[np.cos(dir), np.sin(dir), np.tan(tall)],		 # forward
			[np.cos(dir - wide), np.sin(dir - wide), np.tan(tall)]]) # +wide
		# corresponding points on image
		pts = cv2.fisheye.projectPoints(
			np.expand_dims(pts, 0),
			np.zeros(3).reshape(1, 1, 3),
			np.zeros(3).reshape(1, 1, 3),
			self.K,
			self.D
			)
		return pts[0]

	def fixedProjection(self, i):
		roi = self.fixedRoI(i*self.proj_wide, self.proj_wide, self.proj_long)

		angles = (-self.view_angle, 0, 270-i*self.proj_wide) # alpha, beta, gamma
		# alpha - virtual camera angle
		# gamma - direction 0-360

		R = euler2mat(np.radians(angles))
		pts = cv2.fisheye.undistortPoints(roi, self.K, self.D, R=R, P=None)

		xy = cv2.split(pts)
		xmin, xmax, _, _ = cv2.minMaxLoc(xy[0])
		ymin, ymax, _, _ = cv2.minMaxLoc(xy[1])

		w_out, h_out = self.size_out

		k = np.min([1.*w_out / (xmax-xmin), 1.*h_out / (ymax-ymin)])
		cx = 0.5*(w_out - k*(xmax+xmin));
		cy = 0.5*(h_out - k*(ymax+ymin));
		#cy = - k*ymin;

		P = np.array([ [k  , 0.0, cx], [0.0, k  , cy], [0.0, 0.0, 1.0] ])

		return self.initUndistortRectifyMap(angles, P, self.size_out)

	def get_frame(self, img, i_proj):
		self.image = img
		h,w = self.image.shape[:2]
		W,H = self.size_out

		if self.calibrated:
			M = np.array([ [0.320133,0,0.512611], [0,0.568195,0.48059], [0,0,1] ])
			self.D = np.array([-0.0465311,-0.00820616,0.0210623,-0.0117827])
		else:
			M = np.array([ [0.5,0,0.5], [0,0.5,0.5], [0,0,1] ])
			self.D = np.array([0.0,0.0,0.0,0.0])

		self.K = np.dot(np.matrix([[w,0,0],[0,h,0],[0,0,1]]), M)
		self.P = np.dot(np.matrix([[W,0,0],[0,H,0],[0,0,1]]), M)

		self.num_fixed = 360/self.proj_wide
		self.num_projections = self.num_fixed + 2

		mapx, mapy = self.fixedProjection(i_proj)

		mapx[mapx > w] = -1
		mapy[mapy > h] = -1

		img = self.image.copy()

		img_u = cv2.remap(
		    img,
		    mapx.astype(np.float32),
		    mapy.astype(np.float32),
		    interpolation=cv2.INTER_CUBIC,
		    borderMode=cv2.BORDER_CONSTANT
		)

		return img_u, mapx, mapy
