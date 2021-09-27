package com.hitices.basicenvstatusinfer.client;

import com.hitices.basicenvstatusinfer.core.C45Core;
import com.hitices.basicenvstatusinfer.model.AttributeNode;
import com.hitices.basicenvstatusinfer.utils.C45Utils;
import com.hitices.basicenvstatusinfer.utils.DecisionTreeUtils;
import lombok.Getter;
import org.apache.commons.lang3.math.NumberUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class C45Client {

    private static C45Client c45Client = null;

    @Getter
    private AttributeNode rootNode = null;

    @Getter
    private C45Core core = null;

    public static C45Client getC45Client(String file) throws IOException {
        if(c45Client == null){
            c45Client = new C45Client();
            c45Client.start();
        }
        return c45Client;
    }

    public static void main(String[] args) throws IOException {
        C45Client.getC45Client("file");
//        List<String> featureList = new ArrayList<>(Arrays.asList("OutLook","Humidity","PlayTennis"));
//        List<String> dataList = new ArrayList<>(Arrays.asList("Sunny","80"));
//        String result = C45Client.getC45Client("file").getAnswer(featureList,dataList);
//        System.out.println(result);

    }
    
    private void start() throws IOException {
        List<List<String>> rawData = DecisionTreeUtils.getTrainingData("./data/test.txt");
        C45Utils.transformContinuouslyVariables(rawData);
//        C45Core core = new C45Core();
        core = new C45Core();
        createDecisionTree(core, rawData);
    }
    
    private void createDecisionTree(C45Core core, List<List<String>> currentData) {
        Map<String, Double> maxIGRatioMap = core.maxInformationGainRatio(currentData);
//        AttributeNode rootNode = new AttributeNode(maxIGRatioMap.keySet().iterator().next());
        rootNode = new AttributeNode(maxIGRatioMap.keySet().iterator().next());
        setAttributeNodeStatus(core, currentData, rootNode);
        DecisionTreeUtils.showDecisionTree(rootNode, "");
    }
    
    /**
     * 设置特征属性节点的分支及子节点
     * 
     * @param core
     * @param currentData
     * @param node
     */
    private void setAttributeNodeStatus(C45Core core, List<List<String>> currentData, AttributeNode node) {
        List<String> attributeBranchList = DecisionTreeUtils.getAttributeBranchList(currentData, node.getAttributeName());
        int attributeIndex = DecisionTreeUtils.getAttributeIndex(currentData.get(0), node.getAttributeName());
        
        for (String attributeBranch : attributeBranchList) {
            List<List<String>> splitAttributeDataList = DecisionTreeUtils.splitAttributeDataList(currentData, attributeBranch, attributeIndex);
            buildDecisionTree(core, attributeBranch, splitAttributeDataList, node);
        }
    }
    
    /**
     * 构建 C4.5 决策树
     * 
     * @param core
     * @param attributeBranch
     * @param splitAttributeDataList
     * @param node
     */
    private void buildDecisionTree(C45Core core, String attributeBranch, List<List<String>> splitAttributeDataList, AttributeNode node) {
        Map<String, Double> maxIGRatioMap = core.maxInformationGainRatio(splitAttributeDataList);
        
        String attributeName = maxIGRatioMap.keySet().iterator().next();
        double maxIG = maxIGRatioMap.get(attributeName);
        if (maxIG == 0.0) {
            List<String> singleLineData = splitAttributeDataList.get(splitAttributeDataList.size() - 1);
            
            AttributeNode leafNode = new AttributeNode(singleLineData.get(singleLineData.size() - 1));
            leafNode.setLeaf(true);
            leafNode.setParentStatus(attributeBranch);
            node.addChildNodes(leafNode);
            return;
        }
        
        AttributeNode attributeNode = getNewAttributeNode(attributeName, attributeBranch, node);
        
        setAttributeNodeStatus(core, splitAttributeDataList, attributeNode);
    }
    
    private AttributeNode getNewAttributeNode(String attributeName, String attributeBranch, AttributeNode node) {
        AttributeNode attributeNode = new AttributeNode(attributeName);
        attributeNode.setParentStatus(attributeBranch);
        node.addChildNodes(attributeNode);
        
        return attributeNode;
    }

    /**
     * 利用决策树预测结果
     * @param featureList 特征列表
     * @param  dataList 数据列表
     *
     * @return
     */
    private String getAnswer(List<String> featureList, List<String> dataList){
        AttributeNode decisionNode = rootNode;
        while (decisionNode != null){
            if(decisionNode.isLeaf()){
                return decisionNode.getAttributeName();
            }
            for (int i = 0; i<featureList.size();i++){
                if(featureList.get(i).equals(decisionNode.getAttributeName())){
                    String data = dataList.get(i);
                    AttributeNode childNode = null;
                    for (AttributeNode cn : decisionNode.getChildNodes()){
                        if (NumberUtils.isCreatable(data)){
                            // 连续型随机变量
                            String s = cn.getParentStatus().substring(1);
                            if(NumberUtils.isCreatable(s)){
                                if(NumberUtils.toDouble(data) > NumberUtils.toDouble(s)){
                                    childNode = cn;
                                    break;
                                }
                            }else {
                                if(NumberUtils.toDouble(data) <= NumberUtils.toDouble(s.substring(1))){
                                    childNode = cn;
                                    break;
                                }
                            }
                        }else {
                            // 离散型随机变量
                            if (cn.getParentStatus().equals(data)){
                                childNode = cn;
                                break;
                            }
                        }
                    }
                    if (childNode == null){
                        return "error";
                    }
                    decisionNode = childNode;
                    break;
                }
            }
        }
        return "error";
    }

}
