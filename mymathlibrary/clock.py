import mymathlibrary as mml

class clock:
    def __init__(self):
      self.startTime()

    def startTime(self):
        self.startedTime=time.time() 
 
    def endTime(self,debug=None):
        delta=time.time()-self.startedTime
        if(delta>1):
            print(f"{delta} secs")    
            if(debug):
                print(debug)
