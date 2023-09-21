namespace RandomGame;

/* Lotery game:
    Players alternate picking numbers from 1 to 9.
    After all 9 numbers are picked, machine picks a random number from 1 to 9.
    Player with the closest number wins.

    Probabilities of machine picking numbers, from 1 to 10 are: 
    [1, 2, 3, 4, 5, 6, 7, 8, 9]/45
    So the optimal strategy is to pick the higher numbers first

    // what is a good game that has a random change in the environment?
    // it must be simple to implement and debug. I will use to test a AI agent.
    // a: 
*/

public class GameAction
{
    int pick;
}

// card game
public class Game
{
    int[] numbers = new int[9];
    public int activePlayer;
    public UInt64 chanceHash;

    public List<GameAction> GetPossibleActions()
    {
        throw new NotImplementedException();
    }

    public float[] GetReward()
    {
        throw new NotImplementedException();
    }

    public GameAction GetRandomAction()
    {
        throw new NotImplementedException();
    }

    public static GameAction DefaultPolicy(Game g)
    {
        throw new NotImplementedException();
    }

    public bool Play(GameAction action)
    {
        throw new NotImplementedException();
    }

    public bool IsValid(GameAction action)
    {
        throw new NotImplementedException();
    }

    public bool IsGameOver()
    {
        throw new NotImplementedException();
    }

    public bool IsTerminal()
    {
        throw new NotImplementedException();
    }
}