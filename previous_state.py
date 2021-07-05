class PeopleState:
    frame = 10


    def __init__(self,bbox_L):
        self.headpose = []
        self.Gaze_L = []
        self.Gaze_R = []
        # present, prev1, prev2
        # [[x,y][x,y][x,y]]
        for i in range(self.frame):
            self.headpose.append([0,0])
            self.Gaze_L.append([0,0])
            self.Gaze_R.append([0,0])
        self.prev_bbox_L =bbox_L
        self.threshhold = 100


    def isTheSame(self, bbox_L):
        
        if abs(self.prev_bbox_L - bbox_L) <self.threshhold:
            return True
        else:
            return False
    def set_bbox_l(self, bbox_l):
        self.prev_bbox_L =bbox_l


    def updateState(self, end_x,end_y, selection):
        if selection =="headpose":
            target = self.headpose
        elif selection =="gaze_l":
            target = self.Gaze_L
        elif selection =="gaze_r":
            target = self.Gaze_R

        for i in range(self.frame-1,0,-1):
            target[i] = target[i-1]
        target[0] = [end_x,end_y]
    
    def applyWeight(self,selection):
        if selection =="headpose":
            target = self.headpose
        elif selection =="gaze_l":
            target = self.Gaze_L
        elif selection =="gaze_r":
            target = self.Gaze_R

        weight =0.8
        for i in range(1, self.frame):
            #endpoint x
            if target[i][0]!=0 or target[i][1] !=0:
                if target[i][0] < target[0][0]:
                    target[i][0] += (target[0][0] - target[i][0]) *weight
                else:
                    target[i][0] -= (target[i][0] - target[0][0]) *weight
                # endpoin y
                if target[i][1] < target[0][1]:
                    target[i][1] += (target[0][1] - target[i][1]) *weight
                else:
                    target[i][1] -= (target[i][1] -target[0][1]) *weight

    
    def CalcAverage(self, selection):
        aver_x = 0
        aver_y = 0
        num = 0
        if selection =="headpose":
            target = self.headpose
        elif selection =="gaze_l":
            target = self.Gaze_L
        elif selection == "gaze_r":
            target = self.Gaze_R


        for xy_value in target:
            if xy_value[0] != 0 or xy_value[1] != 0:
                aver_x += xy_value[0]
                aver_y += xy_value[1]
                num += 1
        return aver_x/num , aver_y/num


    def getEndpointAverage(self,end_x, end_y, selection):
        if selection not in ["headpose", "gaze_l", "gaze_r"]:
            print("ERROR: Check Selection Your selection: ", selection)
            return

        self.updateState(end_x,end_y, selection)
        self.applyWeight(selection)
        
        aver_x, aver_y = self.CalcAverage(selection)
        
        return aver_x, aver_y


        
