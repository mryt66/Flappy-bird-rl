using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class BirdFly : MonoBehaviour
{
    public static bool isPaused = false;

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Fly();
        }
    }

    private void Fly()
    {
        GetComponent<Rigidbody2D>().velocity = Vector2.up * 5.5f;
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {

        GetComponent<Rigidbody2D>().velocity = Vector2.zero;
        PauseGame(); // End Bird Fly
    }

    public static void PauseGame()
    {
        isPaused = true;
        Time.timeScale = 0f;
    }

    private void SaveScore()
    {
        string filePath = Application.persistentDataPath + "/score.txt";
        string scoreText = ManageScore.score.ToString();

        // Write the score to the file
        File.WriteAllText(filePath, scoreText);

        Debug.Log("Score saved to file: " + scoreText);
        Debug.Log("File path: " + filePath);
    }
}
