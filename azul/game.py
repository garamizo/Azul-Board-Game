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
from logic_wrapper import Move, Game, get_observation
from ai_wrapper import MCTS

DELAY_ANIMATION = 250
DELAY_AI_MOVE = 4000
AI_MOVE = pygame.USEREVENT + 1
NOT_SET = -30
NUM_COLORS = 5
NUM_ROWS = 5


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

        self.logic = Game(len(self.isHumanPlayer))
        # self.logic = Game.LoadCustomGame()

        # self.logic.reset()
        obs = get_observation(self.logic)
        self.isHumanPlayer = self.isHumanPlayer[:len(obs['player'])]

        self.screen = pygame.display.set_mode(SIZE_SCREEN)
        self.clock = pygame.time.Clock()
        self.setup(obs)
        # print(obs)

        self.aiEngine = MCTS[Game, Move](self.logic, 0.0)
        # self.aiEngine = [None] * self.numPlayers

        # selected commands
        # self.playLineIdx, self.playIsFirst, self.playMark, \
        #     self.playFactoryIdx = None, None, None, None
        self.move = Move()
        self.move.row = self.move.factoryIdx = self.move.color = NOT_SET
        self.move.colIdx = [NOT_SET] * 5
        print(Game.ToScript(self.logic))

    @property
    def activePlayer(self):
        return self.logic.activePlayer

    @property
    def numPlayers(self):
        return self.logic.numPlayers

    def is_game_over(self):
        return self.logic.IsGameOver()

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
                print(Game.ToScript(self.logic))

            elif event.type == pygame.MOUSEBUTTONDOWN:

                self.timeLastEvent = pygame.time.get_ticks()

                # user clicked active board
                if self.activeBoard.rect.collidepoint(event.pos):
                    locStr, locIdx = self.activeBoard.mouse_callback(event)
                    if locStr == 'row' and self.logic.isRegularPhase:
                        self.move.row = locIdx if self.move.row != locIdx else NOT_SET
                        # print(f"Clicked on row {locIdx}")

                    elif locStr == 'row' and self.logic.isRegularPhase == False:
                        self.move.colIdx[locIdx] = NUM_ROWS \
                            if self.move.colIdx[locIdx] != NUM_ROWS else NOT_SET
                        # print(f"Clicked on FLOOR for row {locIdx}")

                    elif locStr == 'colIdx' and self.logic.isRegularPhase == False:
                        self.move.colIdx[locIdx[0]] = locIdx[1] \
                            if self.move.colIdx[locIdx[0]] != locIdx[1] else NOT_SET
                        # print(f"Clicked on grid {locIdx} {self.move}")

                    mouseHandled = True
                    continue

                color, factoryIdx = None, None
                # user clicked center tiles
                if self.center.rect.collidepoint(event.pos):
                    color = self.center.mouse_callback(event)
                    factoryIdx = self.logic.numFactories
                    mouseHandled = True

                # user clicked a factory
                for i, factory in enumerate(self.factory):
                    if factory.rect.collidepoint(event.pos):
                        color = factory.mouse_callback(event)
                        factoryIdx = i
                        mouseHandled = True
                        break

                # print(f"Clicked {color} from factory {factoryIdx}")
                if color is None or (self.move.color == color and self.move.factoryIdx == factoryIdx):
                    self.move.color = NOT_SET
                    self.move.factoryIdx = NOT_SET
                else:
                    self.move.color = color
                    self.move.factoryIdx = factoryIdx

        # play sound if valid click
        if mouseHandled:
            self.SOUND_SELECT.play()
            self.update_draw_ui()
            # print(self.move)

    def move_is_set(self):

        if self.logic.isRegularPhase:
            return self.move.color != NOT_SET and self.move.factoryIdx != NOT_SET and self.move.row != NOT_SET
        else:
            m = Move(self.move, self.logic)
            for col in m.colIdx:
                if col == NOT_SET:
                    return False
            return True

    def process_AI(self, timeout):
        self.aiEngine = self.aiEngine.SearchNode(self.logic, 4).Item1
        if self.aiEngine.numRolls < 1:
            print("Resetting tree...")
        self.aiEngine.GrowWhile(timeout, 300_000)

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
        # if self.move.is_set() == False:
        # if active player is AI, trigger AI to pick move
        self.process_AI(0.1)
        if not self.activeBoard.isHuman:
            self.helpMessage = f"Please wait for AI's turn"
            self._draw()

            # check if is the first AI
            # if self.logic.activePlayer == self.isHumanPlayer.index(False):
            #     self.move = self.logic.GetGreedyMove()
            #     self.delay(1.5)
            #     print("Greedy move: ", self.move)
            # else:
            t0 = pygame.time.get_ticks()
            self.delay(2)
            idx = self.aiEngine.GetBestActionIdx()
            while self.aiEngine.NumRolls(idx) < 1_000:
                self.process_AI(1)
                # self.aiEngine.GrowWhile(1, 100_000)
                idx = self.aiEngine.GetBestActionIdx()
                print(f"{self.aiEngine.NumRolls(idx)}(total: {self.aiEngine.numRolls})  ",
                      end="", flush=True)
                if pygame.time.get_ticks() - t0 > 10_000:
                    break
            self.move = self.aiEngine.GetBestAction() if self.aiEngine.WinRatio(idx) != 0 \
                else self.logic.GetGreedyMove()

            print(f"AI move: {self.aiEngine.ToString(1)}")
            assert self.logic.IsValid(self.move), "AI tried invalid move"

            self.SOUND_AI_TURN.play()
            self.helpMessage = f""
            self.SOUND_SELECT.play()
            self.update_draw_ui()
            self.delay(100 / 1000)

        if self.move_is_set() == False:
            return

        # correct all fields of move
        self.move = Move(self.move, self.logic)

        if self.logic.IsValid(self.move):
            self.delay(100 / 1000)
        else:
            self.SOUND_WRONG.play()
            print(f"{self.move} is invalid")
            self.move.row = self.move.factoryIdx = self.move.color = NOT_SET
            self.move.colIdx = [NOT_SET] * 5
            return

        # print(f"\t\t{self.move}")

        self.obsOld = deepcopy(get_observation(self.logic))
        if self.activeBoard.isHuman:
            print(
                f"Human {self.obsOld['activePlayer'] + 1} action: {self.move}")

        # print("Before\n", self.logic.get_observation())
        self.logic.Play(self.move)
        # reset controls
        self.move = Move()
        self.move.row = self.move.factoryIdx = self.move.color = NOT_SET
        self.move.colIdx = [NOT_SET] * 5

        obs = get_observation(self.logic)
        if self.logic.IsGameOver():  # obs['isGameOver']:  # end of game
            # draw, add score, animate score, animate winner, sing song, wait for quit
            self.animate_game_over()
            # print("End\n", obs)
        elif obs['roundIdx'] != self.obsOld['roundIdx']:  # new round
            # draw, add score, animate score, animate new round, sing song
            self.animate_new_round()
            # print("After\n", self.logic.get_observation())
            print(self.logic)
        else:  # next player
            self.animate_transition()

        self.setup(obs)

    def animate_game_over(self):
        obs = get_observation(self.logic)

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
        obs = deepcopy(get_observation(self.logic))

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

        obsNew = get_observation(self.logic)
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
        if self.logic.isRegularPhase:
            # highlight selected player line
            if self.move.row != NOT_SET:
                if self.move.row == NUM_ROWS:  # draw floor
                    pygame.draw.rect(self.screen, color_select,
                                     self.activeBoard.rect_floor, width=3)
                else:
                    pygame.draw.rect(self.screen, color_select,
                                     self.activeBoard.rect_line[self.move.row], width=3)

            # highlight selected factory tiles
            if self.move.factoryIdx != NOT_SET:
                factory = self.center if self.move.factoryIdx == self.logic.numFactories \
                    else self.factory[self.move.factoryIdx]

                for content, tile in zip(factory.content, factory.tiles):
                    if content == self.move.color:
                        pygame.draw.rect(self.screen, color_select,
                                         tile.rect, width=3)
                if NUM_COLORS in factory.content:  # 1st move tile
                    try:
                        pygame.draw.rect(self.screen, color_select,
                                         factory.tiles[-1].rect, width=3)
                    except IndexError:
                        print(self.logic)
                        print("\n\tfactory cleared\n")
                        self.factoryIdxOn = False
        else:
            for row, col in enumerate(self.move.colIdx):
                # for i in range(25):
                if col >= 0:
                    # row, col = i // NUM_ROWS, i % NUM_ROWS
                    if col >= NUM_ROWS:
                        pygame.draw.rect(self.screen, color_select,
                                         self.activeBoard.rect_line[row], width=3)
                    else:
                        pygame.draw.rect(self.screen, color_select,
                                         self.activeBoard.rect_grid[row*NUM_ROWS + col], width=3)

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
