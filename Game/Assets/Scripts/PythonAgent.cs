using System.IO;
using System;
using UnityEngine;
using System.Collections.Generic;

[Serializable]
public class GameState
{
    public float rect_y;
    public float rect_y_speed;
    public float pipe_x;
    public float pipe_y;
    public float reward;
    public int score;
    public bool done;
}


public class PythonAgent : MonoBehaviour
{
    private static string generalPath = "C:/FlappyBirdBridge/";
    private string stateFilePath = Path.Combine(generalPath, "state.json");
    private string actionFilePath = Path.Combine(generalPath, "action.json");
    private string isDone = Path.Combine(generalPath, "isDone.json");
    private Rigidbody2D playerRigidbody;
    private bool stop = false;

    private GameObject closestPipe;

    public GameObject trianglePrefab;

    void Start()
    {
        RestoreTriangleStates();

        Application.runInBackground = true;
        Physics.gravity = new Vector3(0, -12f, 0);
        playerRigidbody = GetComponent<Rigidbody2D>();
        FindClosestPipe();
    }

    void Update()
    {
        if (stop)
            return;
        FindClosestPipe();

        int action = ReadAgentAction();

        if (action == -1)
        {
            Debug.Log("Brak akcji");
            return;
        }
        if (action == 1)
        {
            Jump();
        }
        WriteGameState();
    }

    //GameState popState;
    float popY;
    void WriteGameState()
    {
        if (ReadIsDone() == 1)
        {
            Time.timeScale = 0f;
            return;
        }
        float penalty = 0;
        float reward = ManageScore.score;
        if (!BirdFly.isPaused)
        {
            Time.timeScale = 5f;
            Time.fixedDeltaTime = 0.02f / Time.timeScale;
        }
        else
        {
            penalty = -10;
            reward = penalty;
        }

        GameState state = new GameState
        {
            pipe_x = GetClosestPipeX() / 11,
            pipe_y = GetClosestPipeY() / 20,
            rect_y = transform.position.y / 20,
            rect_y_speed = (transform.position.y - popY) / 5,
            reward = reward,
            score = ManageScore.pipesScore,
            done = BirdFly.isPaused
        };
        popY = transform.position.y;
        ManageScore.score = 0;

        //if (popState != null)
        //{ // || popState.rect_y_speed == state.rect_y_speed
        //    if (popState.rect_y == state.rect_y || popState.pipe_x == state.pipe_x || popState.pipe_y == state.pipe_y)
        //    {
        //        Debug.Log("Stan gry siê nie zmieni³");
        //        return;
        //    }
        //}

        //popState = state;


        try
        {
            string json = JsonUtility.ToJson(state);
            File.WriteAllText(stateFilePath, json);
            File.Create(isDone).Dispose();

            if (BirdFly.isPaused)
            {
                Debug.Log("Zatrzymano grê");
                RestartGame();
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("Nie uda³o siê zapisaæ stanu gry: " + e.Message);
            return;
        }
    }

    List<GameObject> triangles = new List<GameObject>();
    private List<Vector3> trianglePositions = new List<Vector3>();
    private List<Color> triangleColors = new List<Color>();

    void SaveTriangleStates()
    {
        CombineAndColorTriangles();

        trianglePositions.Clear();
        StaticClass.trianglePositions.Clear();
        trianglePositions = new List<Vector3>();
        StaticClass.trianglePositions = new List<Vector3>();
        StaticClass.triangleColors = new List<Color>();
        foreach (GameObject triangle in triangles)
        {
            trianglePositions.Add(triangle.transform.position);
            StaticClass.trianglePositions.Add(triangle.transform.position);
            triangleColors.Add(triangle.GetComponent<SpriteRenderer>().color);
            StaticClass.triangleColors.Add(triangle.GetComponent<SpriteRenderer>().color);
        }
    }

    void RestoreTriangleStates()
    {
        triangles.Clear();
        trianglePositions = StaticClass.trianglePositions;
        triangleColors = StaticClass.triangleColors;
        int i = 0;
        foreach (Vector3 position in trianglePositions)
        {
            GameObject newTriangle = Instantiate(trianglePrefab, position, Quaternion.identity); // Upewnij siê, ¿e masz prefab do odtworzenia
            newTriangle.GetComponent<SpriteRenderer>().color = triangleColors[i];
            triangles.Add(newTriangle);
            triangleColors.Add(newTriangle.GetComponent<SpriteRenderer>().color);
            i++;
        }
    }



    void CombineAndColorTriangles()
    {
        float combineDistance = 0.3f;
        for (int i = 0; i < triangles.Count; i++)
        {
            for (int j = i + 1; j < triangles.Count; j++)
            {
                float distance = Vector3.Distance(triangles[i].transform.position, triangles[j].transform.position);
                if (distance < combineDistance)
                {
                    Vector3 combinedPosition = (triangles[i].transform.position + triangles[j].transform.position) / 2;

                    Destroy(triangles[i]);
                    Destroy(triangles[j]);

                    GameObject newTriangle = Instantiate(trianglePrefab, combinedPosition, Quaternion.identity);

                    Color newColor = (triangleColors[i].r + -triangleColors[i].g) < (triangleColors[j].r + -triangleColors[j].g) ? triangleColors[j] : triangleColors[i];


                    newColor.r = Mathf.Min(newColor.r + 0.05f, 1f);

                    if (newColor.r >= 1f)
                    {
                        newColor.g = Mathf.Max(newColor.g - 0.05f, 0f);
                    }

                    newTriangle.GetComponent<SpriteRenderer>().color = newColor;

                    triangles[i] = newTriangle;
                    triangleColors[i] = newColor;

                    triangles.RemoveAt(j);
                    triangleColors.RemoveAt(j);
                    j--;
                }
            }
        }
    }


    public void RestartGame()
    {
        //if (ManageScore.pipesScore > 0)//StaticClass.episode > 500)
        if (StaticClass.episode > 500)
        {
            stop = true;
            return;

        }

        float deathY = transform.position.y;

        Debug.Log("Restart gry");
        ManageScore.pipesScore = 0;
        ManageScore.score = 0;
        BirdFly.isPaused = false;
        Time.timeScale = 1f;
        UnityEngine.SceneManagement.SceneManager.LoadScene(0);

        //WaitForSeconds wait = new WaitForSeconds(0.1f);

        GameObject newTriangle = Instantiate(trianglePrefab, new Vector3(0, deathY, -1), Quaternion.identity);
        triangles.Add(newTriangle);
        SaveTriangleStates();

        //List<GameObject> triangles = new List<GameObject>();
        //for (int i = 0; i < StaticClass.episode; i++)
        //{
        //    GameObject newTriangle = Instantiate(triangle, new Vector3(0, 0, -1), Quaternion.identity);
        //    triangles.Add(newTriangle);
        //    Debug.Log("Stworzono trójk¹t");
        //    DontDestroyOnLoad(newTriangle);
        //}
        //for (int i = 0; i < StaticClass.episode; i++)
        //{

        //    foreach (var newTriangle in triangles)
        //    {
        //        if (newTriangle.GetComponent<DeathStats>().id == i)
        //        {
        //            newTriangle.GetComponent<DeathStats>().deathY = deathY;
        //            newTriangle.transform.position = new Vector3(0, deathY, 0);
        //            //bird.GetComponent<DeathStats>().deathSpeed = bird.GetComponent<Rigidbody2D>().velocity.y;
        //            //bird.GetComponent<DeathStats>().deathJumps = bird.GetComponent<PlayerController>().jumps;
        //            //bird.GetComponent<DeathStats>().deathX = bird.transform.position.x;
        //            //bird.GetComponent<DeathStats>().id = i + 1;
        //            break;
        //        }
        //    }
        //}
        StaticClass.episode++;
        Debug.Log(StaticClass.episode);

    }

    //public void StopEverything()
    //{

    //   Time.timeScale = 0f;
    //    BirdFly.isPaused = true;
    //}

    int ReadIsDone()
    {
        if (!File.Exists(isDone))
            return 0;

        return 1;
    }

    int ReadAgentAction()
    {
        if (!File.Exists(actionFilePath))
        {
            return -1;
        }

        try
        {
            string json = File.ReadAllText(actionFilePath);
            var actionData = JsonUtility.FromJson<ActionData>(json);
            return actionData.action;
        }
        catch (Exception e)
        {
            return -1;
        }
    }

    void Jump()
    {
        playerRigidbody.velocity = new Vector2(0, 4.5f);
        //gameObject.transform.position += Vector3.up * 0.5f;
    }

    void FindClosestPipe()
    {
        GameObject[] pipes = GameObject.FindGameObjectsWithTag("Pipe");
        float minDistance = float.MaxValue;
        //GameObject closestPipe = null;
        //Debug.Log("Znaleziono " + pipes.Length + " rur");
        foreach (var pipe in pipes)
        {
            float distance = pipe.transform.position.x - gameObject.transform.position.x;
            if (distance < minDistance && distance > 0)
            {
                minDistance = distance;
                closestPipe = pipe;
            }
        }

        //closestPipe = closestPipe;
    }

    float GetClosestPipeX()
    {
        return closestPipe.transform.position.x - gameObject.transform.position.x;
        //return 200f; // Implementacja logiki dla rury
    }

    float GetClosestPipeY()
    {
        return closestPipe.transform.position.y - gameObject.transform.position.y;
        //return 150f; // Implementacja logiki dla luki
    }

    [Serializable]
    private class ActionData
    {
        public int action;
    }
}
