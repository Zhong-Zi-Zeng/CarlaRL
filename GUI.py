import numpy as np
import pygame




class GUI:
    def __init__(self,win_height=700,win_width=800):
        pygame.init()
        pygame.display.set_caption("Carla")

        # 主要視窗
        self.win_height = win_height
        self.win_width = win_width
        self.win_screen = pygame.display.set_mode((self.win_width, self.win_height))

        # 資訊視窗
        self.info_rect = pygame.Rect(self.win_height, 0, self.win_width, 100)
        pygame.draw.rect(self.win_screen, (0, 0, 0), self.info_rect)

    """繪製遊戲影像"""
    def draw_image(self,bgr_frame):
        image_surface = pygame.surfarray.make_surface(bgr_frame[:,:,::-1].swapaxes(0, 1))
        self.win_screen.blit(image_surface,(0,0))

    """繪製遊戲資訊欄"""
    def draw_text_info(self,text_info,**kwargs):
        text_info.update(kwargs)
        pos = [[10,610],[300,610],[590,610],
               [10,640],[300,640],[590,640],
               [10,670]]

        # 速度、目標點距離、目標點角度、目前輸出命令、訓練次數
        for (key,value),(x,y) in zip(text_info.items(), pos):
            font = pygame.font.Font(None, 30)
            fontSurface = font.render(f'{key}: {value}', True, (255,255,255))
            self.win_screen.blit(fontSurface,(x,y))

    """清空畫面"""
    def clear(self):
        self.win_screen.fill((0,0,0))

    """偵測按鍵事件"""
    def should_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False






