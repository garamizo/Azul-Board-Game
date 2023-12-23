from pygame.image import load
import pygame


def load_sprite(name, with_alpha=True):
    if name == '':
        surface = pygame.Surface((100, 100), pygame.SRCALPHA)
        surface.set_alpha(0)
        return surface
    path = f"assets/sprites/{name}"
    loaded_sprite = load(path)

    if with_alpha:
        return loaded_sprite.convert_alpha()
    else:
        return loaded_sprite.convert()
