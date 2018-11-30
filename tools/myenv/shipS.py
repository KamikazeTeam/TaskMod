import pygame, random, gym, numpy, json, easydict, cv2, time
BLACK, WHITE, DARKGRAY, GRAY, BRIGHTGRAY = (  0,   0,   0),(255, 255, 255),(39,39,39),(192,192,192),(111,111,111)
BLACK, WHITE, DARKGRAY, GRAY, BRIGHTGRAY = list(BLACK), list(WHITE), list(DARKGRAY), list(GRAY), list(BRIGHTGRAY)
class ShipS(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 20}
    def __init__(self):
        super().__init__()
        with open('./myenv/envinfo.json', 'r') as envinfo_file:
            envinfo_args_dict = json.load(envinfo_file)
        args = easydict.EasyDict(envinfo_args_dict)
        self.setsize(args.times,0)
    def setsize(self,times,timelag):
        with open('./myenv/envinfo.json', 'r') as envinfo_file:
            envinfo_args_dict = json.load(envinfo_file)
        args = easydict.EasyDict(envinfo_args_dict)
        self.oneunit, self.timelag = times, timelag
        self.generatefreq,self.endcondition = args.generatefreq, args.endcondition
        self.screen_width,self.screen_height= args.screen_width, args.screen_height
        self.debrissizex, self.debrissizey  = args.debrissizex , args.debrissizey
        self.playersizex, self.playersizey  = args.playersizex , args.playersizey
        self.playerpostx, self.playerposty  = args.playerpostx , args.playerposty
        self.updatespeed, self.actionspeed  = args.updatespeed , args.actionspeed
        self.action_space      = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=(self.screen_width,self.screen_height,1))
        self.reward_range      = [0, 1]
        self._reset()
    def _reset(self):
        self.ocean = numpy.zeros([self.screen_width,self.screen_height,3],numpy.int32)
        self.ship  = [self.playerpostx,self.playerposty]
        self.ocean[self.ship[0]][self.ship[1]]=BRIGHTGRAY
        self.debrislist = []
        self.done, self.counter, self.number, self.outnumber, self.totalnumber = False, 0, 0, 0, 0
        pygame.QUIT
        pygame.init()
        self.clock  = pygame.time.Clock()
        self.screen = pygame.display.set_mode([self.screen_width*self.oneunit, self.screen_height*self.oneunit])
        return self.ocean

    def _step(self, action):
        #if   action==1 and self.ship[0] >                                   0: self.ship[0] -= self.actionspeed
        #elif action==2 and self.ship[0] <  self.screen_width-self.playersizex: self.ship[0] += self.actionspeed
        #elif action==3 and self.ship[1] >                                   0: self.ship[1] -= self.actionspeed
        #elif action==4 and self.ship[1] < self.screen_height-self.playersizey: self.ship[1] += self.actionspeed
        if action==1:
            if self.ship[0] > 0: self.ship[0] -= self.actionspeed
            if self.ship[1] > 0: self.ship[1] -= self.actionspeed
        if action==2:
            if self.ship[0] > 0: self.ship[0] -= self.actionspeed
        if action==3:
            if self.ship[0] > 0: self.ship[0] -= self.actionspeed
            if self.ship[1] < self.screen_height-self.playersizey: self.ship[1] += self.actionspeed
        if action==4:
            if self.ship[1] > 0: self.ship[1] -= self.actionspeed
        if action==5:
            if self.ship[1] < self.screen_height-self.playersizey: self.ship[1] += self.actionspeed
        if action==6:
            if self.ship[0] <  self.screen_width-self.playersizex: self.ship[0] += self.actionspeed
            if self.ship[1] > 0: self.ship[1] -= self.actionspeed
        if action==7:
            if self.ship[0] <  self.screen_width-self.playersizex: self.ship[0] += self.actionspeed
        if action==8:
            if self.ship[0] <  self.screen_width-self.playersizex: self.ship[0] += self.actionspeed
            if self.ship[1] < self.screen_height-self.playersizey: self.ship[1] += self.actionspeed

        reward = 0
        if self.ship in self.debrislist:
            self.debrislist.remove(self.ship)
            self.number+=1
            reward = 1

        self.outnumber=0
        for debris in self.debrislist:
            if debris[1] >= self.screen_height+self.actionspeed:
                self.outnumber+=1
            else:
                debris[1]+=1
        #if self.outnumber>=self.endcondition: self.done=True

        if self.counter%self.generatefreq==0:
            newdebris=[random.randrange(self.screen_width),0]
            self.debrislist.append(newdebris)
            self.totalnumber+=1
        if self.totalnumber-self.number>=self.endcondition: self.done=True

        self.counter+=1

        self.ocean*=0
        self.ocean[self.ship[0]][self.ship[1]]=BRIGHTGRAY
        for debris in self.debrislist:
            try:
                self.ocean[debris[0]][debris[1]]=WHITE
            except:
                pass

        return self.ocean, reward, self.done, {}

    def _render(self, mode='human', close=False):
        #print(self.number,self.outnumber)
        frame = self.ocean#*123#numpy.zeros([self.screen_width,self.screen_height,3])
        #for i in range(len(self.ocean)):
        #    for j in range(len(self.ocean[i])):
        #        if self.ocean[i][j] == 0: frame[i][j]=[0,0,0]
        #        if self.ocean[i][j] == 1: frame[i][j]=[123,123,123]
        #        if self.ocean[i][j] == 2: frame[i][j]=[234,234,234]
        frame = cv2.resize(frame,None,fx=self.oneunit,fy=self.oneunit,interpolation=cv2.INTER_NEAREST)
        surf  = pygame.surfarray.make_surface(frame)
        self.screen.blit(surf, (0, 0))
        pygame.display.update()
        self.clock.tick(self.timelag)
        return pygame.surfarray.array3d(self.screen).swapaxes(0,1)
    def _close(self):
        pygame.quit()
    def _seed(self, seed=None):
        random.seed(seed)
