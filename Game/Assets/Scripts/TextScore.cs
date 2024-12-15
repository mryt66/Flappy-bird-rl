using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class TextScore : MonoBehaviour
{
    public static float score = 0;

    // Update is called once per frame
    void Update()   
    {
        score = ManageScore.pipesScore;
        gameObject.GetComponent<TextMeshProUGUI>().text = "Score: " + score.ToString();
    }
}
