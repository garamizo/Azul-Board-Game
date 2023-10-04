import pygame
from models import GameObject, Board, Factory, Tile, Center
from models import SIZE_SCREEN, SIZE_BOARD, SIZE_FACTORY, SIZE_TILE
from models import BLACK, BEIGE, WHITE, YELLOW, RED
from pygame.math import Vector2
from copy import deepcopy
from random import seed
from tqdm import tqdm
from logic import print_state

# from logic import Table
# from ai import MCTS_node
from logic_wrapper import Table
from ai_wrapper import MCTS_node

DELAY_ANIMATION = 250
DELAY_AI_MOVE = 4000
AI_MOVE = pygame.USEREVENT + 1
COLOR_NAME = {1: "BLUE", 2: "YELLOW", 3: "RED",
              4: "BLACK", 5: "WHITE", 6: "FIRST"}
ROW_NAME = {-1: "FLOOR", 0: "FIRST", 1: "SECOND",
            2: "THIRD", 3: "FOURTH", 4: "FIFTH"}


class Azul:
    isHumanPlayer = [True, False, False]
    aiEngine = None

    def __init__(self):
        # seed(1)
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
        self.SOUND_HUMAN_WIN = pygame.mixer.Sound(
            'assets/sounds/mixkit-medieval-show-fanfare-announcement-226.wav')
        self.SOUND_AI_WIN = pygame.mixer.Sound(
            'assets/sounds/Bidibodi_bidibu_radio.mp3')
        self.SOUND_AI_TURN = pygame.mixer.Sound(
            'assets/sounds/mixkit-retro-confirmation-tone-2860.wav')

        # messages
        self.FONT_DISPLAY_MESSAGE = pygame.font.SysFont(None, 200, bold=False)
        self.FONT_HELP_MESSAGE = pygame.font.SysFont(None, 40, bold=False)

        self.logic = Table(len(self.isHumanPlayer))
        # self.logic.core = self.logic.core.LoadCustomGame()

        # self.logic.reset()
        obs = self.logic.get_observation()
        self.isHumanPlayer = self.isHumanPlayer[:len(obs['player'])]

        self.screen = pygame.display.set_mode(SIZE_SCREEN)
        self.clock = pygame.time.Clock()
        self.setup(obs)
        # print(obs)

        self.aiEngine = MCTS_node(self.logic)
        # self.aiEngine = [None] * self.numPlayers

        # selected commands
        self.playLineIdx, self.playIsFirst, self.playMark, \
            self.playFactoryIdx = None, None, None, None

    @property
    def activePlayer(self):
        return self.logic.activePlayer

    @property
    def numPlayers(self):
        return self.logic.numPlayers

    def is_game_over(self):
        return self.logic.is_game_over()

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
            Board(f'{"Human" if isHuman else "AI"} {i+1}', b, (g + (g+x)*i + x/2, y), size=SIZE_BOARD, zoom=0.6, isHuman=isHuman) if i < p1
            else (Board(f'{"Human" if isHuman else "AI"} {i+1}', b, (g + (g+x)*(i-1) + x/2 + w, y), size=SIZE_BOARD, zoom=0.6, isHuman=isHuman) if i > p1
                  else Board(f'{"Human" if isHuman else "AI"} {i+1}', b, SIZE_BOARD / 2 + Vector2(g, g), size=SIZE_BOARD, zoom=1.0, isHuman=isHuman))
            for i, (b, isHuman) in enumerate(zip(obs['player'], self.isHumanPlayer))
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
        while not self.is_game_over():
            self._handle_input()
            self._process_game_logic()
            self._draw()
            self.process_AI(0.1)

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

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # obs = self.logic.get_observation()
                # print(obs)
                print(self.logic.get_printable())
            elif event.type == pygame.MOUSEBUTTONDOWN:

                self.timeLastEvent = pygame.time.get_ticks()

                # user clicked active board
                if self.activeBoard.rect.collidepoint(event.pos):
                    playLineIdx = self.activeBoard.mouse_callback(event)
                    self.playLineIdx = playLineIdx if self.playLineIdx != playLineIdx else None
                    mouseHandled = True
                    continue

                # user clicked center tiles
                if self.center.rect.collidepoint(event.pos):
                    mouseHandled = True
                    playMark, _, playIsFirst = \
                        self.center.mouse_callback(event)
                    if self.playMark != playMark or self.playIsFirst != playIsFirst:
                        self.playMark = playMark
                        self.playIsFirst = playIsFirst
                    else:
                        self.playMark = self.playIsFirst = None

                    self.playFactoryIdx = -1
                    continue

                # user clicked a factory
                for i, factory in enumerate(self.factory):
                    if factory.rect.collidepoint(event.pos):
                        mouseHandled = True
                        playMark, _ = factory.mouse_callback(event)
                        playFactoryIdx = i
                        if self.playMark != playMark or self.playFactoryIdx != playFactoryIdx:
                            self.playMark, self.playFactoryIdx = playMark, playFactoryIdx
                        else:
                            self.playMark, self.playFactoryIdx = None, None

                        break

            elif event.type == AI_MOVE:
                mouseHandled = True
                self.helpMessage = f"Please wait for AI's turn"
                self._draw()

                # t0 = pygame.time.get_ticks()
                # MIN_ROLLS, rolls = 4000, 0
                # with tqdm(total=MIN_ROLLS) as pbar:
                # while pygame.time.get_ticks() - t0 < 4_000:
                self.process_AI(4)
                # numRolls = self.aiEngine[self.activePlayer].numRolls
                # pbar.update(numRolls - rolls)
                # rolls = numRolls

                # actionIdx = self.aiEngine[self.activePlayer].get_best_action()

                # if current player is first of the non-human players
                # if self.isHumanPlayer.index(False) == self.activePlayer:
                #     self.playFactoryIdx, self.playMark, self.playLineIdx = \
                #         self.logic.get_egreedy_move(0.0)
                #     numRolls, numWins = 1, -1
                # else:
                self.playFactoryIdx, self.playMark, self.playLineIdx, \
                    numRolls, numWins = self.aiEngine.get_action()

                print((f"AI {self.logic.activePlayer + 1} action:"
                       f"factoryIdx: {self.playFactoryIdx}, "
                       f"color: {COLOR_NAME[self.playMark]}, row: {ROW_NAME[self.playLineIdx]}, "
                       f"numSims: {numRolls}, "
                       f"avgReward: {numWins/numRolls: .3}"))

                if self.logic.is_valid(self.playMark, self.playFactoryIdx, self.playLineIdx) == False:
                    print(
                        f"AI {self.logic.activePlayer + 1} tried an invalid move")
                    self.playLineIdx = None
                    self.logic.core.Print()

                    self.aiEngine[self.activePlayer].core.state.Print()
                    assert False, "AI tried invalid move"

                self.SOUND_AI_TURN.play()
                self.helpMessage = f""

        # play sound if valid click
        if mouseHandled:
            self.SOUND_SELECT.play()
            self.update_draw_ui()

        # test if move is valid
        if (mouseHandled and self.playLineIdx is not None and self.playMark is not None):

            self.update_draw_ui()
            if self.logic.is_valid(self.playMark, self.playFactoryIdx, self.playLineIdx):
                # pygame.time.delay(100)
                self.delay(100 / 1000)
            else:
                self.SOUND_WRONG.play()
                print("Invalid move")

    def process_AI(self, timeout):
        # numAIPlayers = len(
        #     [1 for isHuman in self.isHumanPlayer if isHuman == False])
        # if numAIPlayers == 0:
        #     return

        # for i, isHuman in enumerate(self.isHumanPlayer):
        #     if isHuman:
        #         continue

        #     # load for the first time
        #     if self.aiEngine[i] is None:
        #         self.aiEngine[i] = MCTS_node(self.logic)
        #         self.aiEngine[i].core.rootPlayer = i

        #     self.aiEngine[i] = self.aiEngine[i].search_node(
        #         self.logic, maxDepth=self.logic.numPlayers)

        self.aiEngine = self.aiEngine.search_node(self.logic, maxDepth=4)
        self.aiEngine.grow_while(
            timeout=timeout, maxRolls=50_000)

    def delay(self, secs):
        t0 = pygame.time.get_ticks()
        self.process_AI(secs)
        pygame.time.delay(int(secs * 1000) - (pygame.time.get_ticks() - t0))

    def wait_for_click(self):
        # return
        while True:
            self.process_AI(0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    return

    def _process_game_logic(self):

        # move not selected yet
        if (self.playLineIdx is None or self.playMark is None):
            # if active player is AI, trigger AI to pick move
            if not self.activeBoard.isHuman:
                pygame.event.post(pygame.event.Event(AI_MOVE))
            return
        # return if move is invalid
        if not self.logic.is_valid(self.playMark, self.playFactoryIdx, self.playLineIdx):
            return

        self.obsOld = deepcopy(self.logic.get_observation())
        if self.activeBoard.isHuman:
            print((f"Human {self.obsOld['activePlayer'] + 1} action:"
                   f"factoryIdx: {self.playFactoryIdx}, "
                   f"color: {COLOR_NAME[self.playMark]}, row: {ROW_NAME[self.playLineIdx]}"))

        # print("Before\n", self.logic.get_observation())
        self.logic.step_move(
            self.playMark, self.playFactoryIdx, self.playLineIdx)

        obs = self.logic.get_observation()
        if self.logic.is_game_over():  # obs['isGameOver']:  # end of game
            # draw, add score, animate score, animate winner, sing song, wait for quit
            self.animate_game_over()
            # print("End\n", obs)
        elif obs['roundIdx'] != self.obsOld['roundIdx']:  # new round
            # draw, add score, animate score, animate new round, sing song
            self.animate_new_round()
            # print("After\n", self.logic.get_observation())
            print(self.logic.get_printable())
        else:  # next player
            self.animate_transition()

        self.setup(obs)

        # reset controls
        self.playLineIdx, self.playIsFirst, self.playMark, self.playFactoryIdx = \
            None, None, None, None

    def animate_game_over(self):
        obs = self.logic.get_observation()

        pygame.time.delay(DELAY_ANIMATION * 5)
        self.displayMessage = f"Game Over"

        for ip in range(len(self.board)):
            self.SOUND_SCORE.play()
            # display old + new score
            diffScore = obs['player'][ip]['score'] - \
                self.obsOld['player'][ip]['score']
            self.board[ip].set_score(
                f"{self.obsOld['player'][ip]['score']} + {diffScore} = {obs['player'][ip]['score']}")
            self._draw()
            pygame.time.delay(DELAY_ANIMATION * 4)

        pygame.time.delay(DELAY_ANIMATION * 2)

        allScores = [p['score'] for p in obs['player']]
        winnerIdx, _ = max(enumerate(allScores), key=lambda x: x[1])
        self.displayMessage = f"{self.board[winnerIdx].name} Wins"
        self.helpMessage = f"Click anywhere to quit"
        self._draw()

        (self.SOUND_HUMAN_WIN if self.isHumanPlayer[winnerIdx] else self.SOUND_AI_WIN).play(
        )
        self.wait_for_click()
        print("Game over. Closing...")
        # quit()

    def animate_new_round(self):
        self.setup(self.obsOld)
        obs = deepcopy(self.logic.get_observation())

        # pygame.time.delay(DELAY_ANIMATION * 5)
        self.delay(DELAY_ANIMATION * 5 / 1000)

        self.displayMessage = f"Round {obs['roundIdx'] + 1}"

        for ip in range(len(self.board)):
            self.SOUND_SCORE.play()
            self.board[ip].set_content(obs['player'][ip])
            # display old + new score
            diffScore = obs['player'][ip]['score'] - \
                self.obsOld['player'][ip]['score']
            self.board[ip].set_score(
                f"{self.obsOld['player'][ip]['score']} + {diffScore} = {obs['player'][ip]['score']}")
            self._draw()
            # pygame.time.delay(DELAY_ANIMATION * 4)
            self.delay(DELAY_ANIMATION * 4 / 1000)

        self.helpMessage = f"Click anywhere to continue"
        self._draw()
        self.wait_for_click()
        self.SOUND_SWITCH.play()
        self.displayMessage = f""
        self.helpMessage = f""

    def animate_transition(self):

        obsNew = self.logic.get_observation()
        obsNew['activePlayer'] = self.obsOld['activePlayer']

        for i in range(3):
            obs = self.obsOld if i % 2 else obsNew
            self.setup(obs)
            self._draw()
            # pygame.time.delay(DELAY_ANIMATION)
            self.delay(DELAY_ANIMATION / 1000)

        # pygame.time.delay(DELAY_ANIMATION)
        self.delay(DELAY_ANIMATION / 1000)
        self.SOUND_SWITCH.play()

    def update_draw_ui(self):
        color_select = RED
        # highlight selected player line
        if self.playLineIdx is not None:
            if self.playLineIdx == -1:
                pygame.draw.rect(self.screen, color_select,
                                 self.activeBoard.rect_floor, width=3)
            else:
                pygame.draw.rect(self.screen, color_select,
                                 self.activeBoard.rect_line[self.playLineIdx], width=3)

        # highlight selected factory tiles
        if self.playFactoryIdx is not None:
            factory = self.factory[self.playFactoryIdx] \
                if self.playFactoryIdx >= 0 else self.center

            for content, tile in zip(factory.content, factory.tiles):
                if content == self.playMark:
                    pygame.draw.rect(self.screen, color_select,
                                     tile.rect, width=3)
            if self.playIsFirst:
                try:
                    pygame.draw.rect(self.screen, color_select,
                                     factory.tiles[0].rect, width=3)
                except IndexError:
                    print(self.logic.get_printable())
                    print("\n\tfactory cleared\n")
                    self.playFactoryIdx = None

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


if __name__ == "__main__":
    Azul().main_loop()
