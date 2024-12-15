using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ManageScore : MonoBehaviour
{
    public static float score = 0;
    public static int pipesScore = 0;
    public static float fullScore = 0;

    private void Start()
    {
        fullScore = 0;
    }

    public static void ScoreUp(int points)
    {
        score += points;
        fullScore += points;
    }

    public static void ScoreUp(float points)
    {
        score += points;
        fullScore += points;
    }

    public static void PipesScoreUp(int points)
    {
        pipesScore += points;
    }

}
