using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class IncreaseScore : MonoBehaviour
{
    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (collision.gameObject.tag == "Player")
        {
            ManageScore.PipesScoreUp(1);
            ManageScore.ScoreUp(1);
        }
    }
}
