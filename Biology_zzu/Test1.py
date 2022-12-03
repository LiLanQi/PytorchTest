class Model(object):
    def __init__(self, graph):        """        graph: 我们前面创建好的图        """
    # 创建 GraphWrapper 图数据容器，用于在定义模型的时候使用，后续训练时再feed入真实数据
    # self.gw = pgl.graph_wrapper.GraphWrapper(name='graph',
    # node_feat=graph.node_feat_info(),
    # edge_feat=graph.edge_feat_info())
    # 作用同 GraphWrapper，此处用作节点标签的容器
    # self.node_label = fluid.layers.data("node_label", shape=[None, 1],
    # dtype="float32", append_batch_size=False)

    def build_model(self):
        # 定义两层model_layer
        #  output = model_layer(self.gw,
        #  self.gw.node_feat['feature'],
        #  self.gw.edge_feat['edge_feature'],
        #  hidden_size=8,
        #  name='layer_1',
        #  activation='relu')
        #  output = model_layer(self.gw,
        #  output,
        #  self.gw.edge_feat['edge_feature'],
        #  hidden_size=1,
        #  name='layer_2',
        #  activation=None)
        # 对于二分类任务，可以使用以下 API 计算损失
        # loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=output,                                                               label=self.node_label)        # 计算平均损失        loss = fluid.layers.mean(loss)                # 计算准确率        prob = fluid.layers.sigmoid(output)        pred = prob > 0.5        pred = fluid.layers.cast(prob > 0.5, dtype="float32")        correct = fluid.layers.equal(pred, self.node_label)        correct = fluid.layers.cast(correct, dtype="float32")        acc = fluid.layers.reduce_mean(correct)
        return loss, acc