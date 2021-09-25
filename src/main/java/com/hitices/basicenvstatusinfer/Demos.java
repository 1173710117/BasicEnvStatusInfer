package com.hitices.basicenvstatusinfer;

import com.hitices.basicenvstatusinfer.utils.C45Utils;
import com.hitices.basicenvstatusinfer.utils.CommonUtils;
import com.hitices.basicenvstatusinfer.utils.DecisionTreeUtils;

import java.io.IOException;
import java.util.List;

public class Demos {

    public static void main(String[] args) throws IOException {
//        System.out.println(e(5, 14) + e(4, 14) + e(5, 14));
        
        double info1 = (6.0/14) * (e(4, 6) + e(2, 6));
        double info2 = (8.0/14) * (e(5, 8) + e(3, 8));
        System.out.println(info1 + info2);
        
        List<List<String>> currentData = DecisionTreeUtils.getTrainingData("./data/weather2.txt");
        C45Utils.transformContinuouslyVariables(currentData);
    }
    
    private static double e(int a, int b) {
        return -1.0 * (1.0 * a / b) * CommonUtils.log2(1.0 * a / b);
    }
}
