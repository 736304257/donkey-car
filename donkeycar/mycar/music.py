# -*- coding: utf-8 -*-
import pygame
import time
pygame.mixer.pre_init(44100, -16, 2, 2048) # fix audio delay
pygame.init()
pygame.mixer.music.load("music1.mp3")
pygame.mixer.music.play(-1)
time.sleep(1000)