import pygame
from utils import load_sprite
from models import GameObject, Board, Factory, Tile, Center
from models import SIZE_SCREEN, SIZE_BOARD, SIZE_FACTORY, SIZE_TILE
from models import BLACK, BEIGE, WHITE, YELLOW
from pygame.math import Vector2
from logic import Table
import copy

DELAY_ANIMATION = 200


class Azul:
    def __init__(self):
        self._init_pygame()
        pygame.mixer.init()
        self.timeLastEvent = 0
        self.roundIdx = -1
        self.displayMessage = f""
        self.helpMessage = f""

        # load sounds
        self.SOUND_SELECT = pygame.mixer.Sound(
            'assets/sounds/mixkit-poker-card-placement-2001.wav')
        self.SOUND_WRONG = pygame.mixer.Sound(
            'assets/sounds/mixkit-video-game-mystery-alert-234.wav')
        self.SOUND_SWITCH = pygame.mixer.Sound(
            'assets/sounds/mixkit-paper-slide-1530.wav')
        self.SOUND_SCORE = pygame.mixer.Sound(
            'assets/sounds/mixkit-small-win-2020.wav')
        self.SOUND_OVER = pygame.mixer.Sound(
            'assets/sounds/mixkit-medieval-show-fanfare-announcement-226.wav')

        # messages
        self.FONT_DISPLAY_MESSAGE = pygame.font.SysFont(None, 200, bold=False)
        self.FONT_HELP_MESSAGE = pygame.font.SysFont(None, 40, bold=False)

        self.logic = Table()
        # self.logic.reset()
        obs = self.logic.get_observation()

        self.screen = pygame.display.set_mode(SIZE_SCREEN)
        self.clock = pygame.time.Clock()
        self.setup(obs)

        # selected commands
        self.playLineIdx, self.playIsFirst, self.playMark, \
            self.playNumMark, self.playFactoryIdx = None, None, None, None, None

    def setup(self, obs):

        numPlayers, numFactories = len(obs['player']), len(obs['factory'])
        sb, sf, s = SIZE_BOARD, SIZE_FACTORY, SIZE_SCREEN
        st = SIZE_TILE[0]
        ssb = SIZE_BOARD * 0.6
        g, x = 10, SIZE_BOARD[0] * 0.6
        w = s[0] - numPlayers*g - (numPlayers-1)*x
        y = s[1] - ssb[1]/2 - g
        p1 = obs['activePlayer']

        # position boards
        self.board = [
            Board(f'Player {i+1}', b, (g + (g+x)*i + x/2, y), size=SIZE_BOARD, zoom=0.6) if i < p1
            else (Board(f'Player {i+1}', b, (g + (g+x)*(i-1) + x/2 + w, y), size=SIZE_BOARD, zoom=0.6) if i > p1
                  else Board(f'Player {i+1}', b, SIZE_BOARD / 2 + Vector2(g, g), size=SIZE_BOARD, zoom=1.0))
            for i, b in enumerate(obs['player'])
        ]
        self.activeBoard = self.board[p1]

        # position factories
        x1, x2 = sb.x + sf.x/2 + 1.5*g, s.x - sf.y/2 - 0.5*g
        y1, y2 = (sb.y + 2*g)/2 - (sf.y+2*st+g)/2 - g, \
            (sb.y + 2*g)/2 + (sf.y+2*st+g)/2 + g
        self.factory = [
            Factory(f, (x1 + (x2-x1)*i/4, y1), size=SIZE_FACTORY) if i < 5
            else Factory(f, (x1 + (x2-x1)*(i-5)/4, y2), size=SIZE_FACTORY)
            for i, f in enumerate(obs['factory'])]

        # position center tiles
        sz = Vector2(s.x - sb.x - 3*g, 2*st + 2*g)
        s0 = Vector2(sb.x + 2*g + sz.x/2, sb.y/2 + g)
        self.center = Center(obs['center'], s0, sz)

    def main_loop(self):
        while True:
            self._handle_input()
            # if pygame.time.get_ticks() - self.timeLastEvent > 500:
            self._process_game_logic()
            self._draw()

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Azul")

    def _handle_input(self):

        mouseHandled = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                self.timeLastEvent = pygame.time.get_ticks()

                if self.activeBoard.rect.collidepoint(event.pos):
                    playLineIdx = self.activeBoard.mouse_callback(event)
                    self.playLineIdx = playLineIdx if self.playLineIdx != playLineIdx else None
                    mouseHandled = True
                    continue

                if self.center.rect.collidepoint(event.pos):
                    mouseHandled = True
                    playMark, playNumMark, playIsFirst = \
                        self.center.mouse_callback(event)
                    if self.playMark != playMark or self.playNumMark != playNumMark or self.playIsFirst != playIsFirst:
                        self.playMark = playMark
                        self.playNumMark = playNumMark
                        self.playIsFirst = playIsFirst
                    else:
                        self.playMark = self.playNumMark = self.playIsFirst = None

                    self.playFactoryIdx = -1
                    continue

                for i, factory in enumerate(self.factory):
                    if factory.rect.collidepoint(event.pos):
                        mouseHandled = True
                        playMark, playNumMark = factory.mouse_callback(event)
                        playFactoryIdx = i
                        if self.playMark != playMark or self.playNumMark != playNumMark \
                                or self.playFactoryIdx != playFactoryIdx:
                            self.playMark, self.playNumMark, self.playFactoryIdx = playMark, playNumMark, playFactoryIdx
                        else:
                            self.playMark, self.playNumMark, self.playFactoryIdx = None, None, None

                        break

        # and (self.playLineIdx is not None or self.playMark is not None):
        if mouseHandled:
            self.SOUND_SELECT.play()
            self.update_draw_ui()

        # test if move is valid
        if (mouseHandled and self.playLineIdx is not None and self.playMark is not None):

            self.update_draw_ui()
            if self.logic.is_valid(self.playMark, self.playFactoryIdx, self.playLineIdx):
                pygame.time.delay(100)

                print("Switch player")
            else:
                self.SOUND_WRONG.play()
                print("Invalid move")

    def wait_for_click(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    return

    def _process_game_logic(self):

        # return if move is not selected
        if (self.playLineIdx is None or self.playMark is None):
            return
        # return if move is invalid
        if not self.logic.is_valid(self.playMark, self.playFactoryIdx, self.playLineIdx):
            return

        self.obsOld = copy.deepcopy(self.logic.get_observation())
        self.logic.step_move(
            self.playMark, self.playFactoryIdx, self.playLineIdx)

        obs = self.logic.get_observation()
        if obs['isGameOver']:  # end of game
            # draw, add score, animate score, animate winner, sing song, wait for quit
            self.animate_game_over()
        elif obs['roundIdx'] != self.obsOld['roundIdx']:  # new round
            # draw, add score, animate score, animate new round, sing song
            self.animate_new_round()

        else:  # next player
            self.animate_transition()

        self.setup(obs)

        # reset controls
        self.playLineIdx, self.playIsFirst, self.playMark, self.playNumMark, self.playFactoryIdx = \
            None, None, None, None, None

    def animate_game_over(self):
        obs = self.logic.get_observation()

        self.displayMessage = f"Game Over"

        for ip in range(len(self.board)):
            self.SOUND_SCORE.play()
            # display old + new score
            diffScore = obs['player'][ip]['score'] - \
                self.obsOld['player'][ip]['score']
            self.board[ip].set_score(
                f"{self.obsOld['player'][ip]['score']} + {diffScore} = {obs['player'][ip]['score']}")
            self._draw()
            pygame.time.delay(DELAY_ANIMATION * 5)

        allScores = [p['score'] for p in obs['player']]
        winnerIdx, _ = max(enumerate(allScores), key=lambda x: x[1])
        self.displayMessage = f"{self.board[winnerIdx].name} Wins"
        self.helpMessage = f"Click anywhere to quit"
        self._draw()

        self.SOUND_OVER.play()
        self.wait_for_click()
        quit()

    def animate_new_round(self):
        obs = self.logic.get_observation()

        self.displayMessage = f"Round {obs['roundIdx'] + 1}"
        self.helpMessage = f"Click anywhere to continue"

        for ip in range(len(self.board)):
            self.SOUND_SCORE.play()
            # display old + new score
            diffScore = obs['player'][ip]['score'] - \
                self.obsOld['player'][ip]['score']
            self.board[ip].set_score(
                f"{self.obsOld['player'][ip]['score']} + {diffScore}")
            self._draw()
            pygame.time.delay(DELAY_ANIMATION * 5)

        self.wait_for_click()
        self.displayMessage = f""
        self.helpMessage = f""

    def animate_transition(self):

        obsNew = self.logic.get_observation()
        obsNew['activePlayer'] = self.obsOld['activePlayer']

        for i in range(5):
            obs = self.obsOld if i % 2 else obsNew
            self.setup(obs)
            self._draw()
            pygame.time.delay(DELAY_ANIMATION)

        pygame.time.delay(DELAY_ANIMATION)
        self.SOUND_SWITCH.play()

    def update_draw_ui(self):
        # highlight selected player line
        if self.playLineIdx is not None:
            pygame.draw.rect(self.screen, (255, 0, 0),
                             self.activeBoard.rect_line[self.playLineIdx], width=3)

        # highlight selected factory tiles
        if self.playFactoryIdx is not None:
            factory = self.factory[self.playFactoryIdx] \
                if self.playFactoryIdx >= 0 else self.center

            for content, tile in zip(factory.content, factory.tiles):
                if content == self.playMark:
                    pygame.draw.rect(self.screen, (255, 0, 0),
                                     tile.rect, width=3)
            if self.playIsFirst:
                pygame.draw.rect(self.screen, (255, 0, 0),
                                 factory.tiles[0].rect, width=3)

        imgMessage = self.FONT_DISPLAY_MESSAGE.render(
            self.displayMessage, True, YELLOW, BLACK)
        pos = SIZE_SCREEN/2 - Vector2(imgMessage.get_size())/2
        self.screen.blit(imgMessage, pos)

        imgHelp = self.FONT_HELP_MESSAGE.render(
            self.helpMessage, True, WHITE, BLACK)
        pos = SIZE_SCREEN - Vector2(imgHelp.get_size()) - Vector2(10, 10)
        self.screen.blit(imgHelp, pos)

        pygame.display.flip()

    def _draw(self):

        self.screen.fill((0, 0, 0))

        for board in self.board:
            board.draw(self.screen)

        for factory in self.factory:
            factory.draw(self.screen)

        self.center.draw(self.screen)

        self.update_draw_ui()

        pygame.display.flip()
        self.clock.tick(20)
