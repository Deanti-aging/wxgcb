wxgcb = function(){
  #ssmd是因为从无监督变为半监督才和其他半监督算法做对比，而这里其实没必要，只要基于knn和ksnn分别变化，而且发现变后比变之前强就够了
  #最后脑龄预测只要出好模型就行
  #和原算法对比，半监督和全监督对比，我的半监督和其他半监督对比
  #真正靠谱的做法：为了评估模型的公平，进行训练/测试集的划分和交叉验证取均值；为了调参/选模型，对训练集进行训练/验证集的划分和交叉验证挑最佳；为了保持原数据结构（尤其是一些受真实簇数k影响的算法），两种做法和目的的交叉验证中的划分折都要分层等比例采样，对于数据集过小的情况就自助法过采样；而且留一法选出的超参数更符合全体数据集的情况
  if (TRUE) {
    # install.packages("dplyr")
    # install.packages("ggplot2")
    # install.packages("tidytext")
    # install.packages("pacman")
    # install.packages("ggthemes")
    # install.packages("readr")
    # install.packages("showtext")
    # install.packages("patchwork")
    # install.packages("cluster")
    # install.packages("kmed")
    # install.packages("conclust")
    # install.packages("clv")
    # install.packages("SSLR")
    # install.packages("dbscan")
    # install.packages("plot3D")
    # install.packages("wesanderson")
    # install.packages("fs")#for make dir
    # install.packages("gensvm")#for svm
    # install.packages("Rdimtools")#for deminsion reduction
    # install.packages("FactoMineR")#for pca
    # install.packages("neuronorm")#for process sMRI
    # install.packages("RNifti")#for read nifti
    # install.packages("smotefamily")#for oversample
    # install.packages("FNN")#for oversample 用k近邻
    # install.packages("igraph")#for oversample
    # install.packages("vioplot")#for vioplot
    # install.packages("apa")#for Cohen's d
    # install.packages("kknn")
    # install.packages("FastKNN")
    # install.packages("ksNN")
    # install.packages("mildsvm")
    # install.packages("ramsvm")
    # install.packages("metR")#for counter plot
    # install.packages("arulesCBA")
    # install.packages("dbstats")
    # install.packages("RSNNS")#神经网络
    # install.packages("RSSL") #半监督学习
    library(RSSL)
    library(RSNNS)
    library(dbstats)
    library(arulesCBA)
    library(metR)
    library(smotefamily)#for oversample
    library(ramsvm)
    library(mildsvm)
    library(ksNN)
    library(FastKNN)
    library(kknn)
    library(Rtsne)#for t-SNE
    library(apa)
    library(vioplot)
    library(FNN)
    library(igraph)
    library(smotefamily)
    library(RNifti)
    library(FactoMineR)
    library(Rdimtools)
    library(gensvm)
    library(wesanderson)
    library(plot3D)
    library(dplyr)
    library(cluster)
    library(ggplot2)
    library(tidytext)
    library(patchwork)
    library(kmed)
    library(conclust)
    library(clv)
    library(SSLR)
    library(pacman)
    library(ggthemes)
    library(readr)
    library(showtext)
    library(dbscan)
    library(fs)
    library(e1071)#svr
  }
  scriptpath = rstudioapi::getSourceEditorContext()$path
  scriptpathn = nchar(scriptpath)
  suppath = substr(scriptpath, 1, scriptpathn-8)#注意这个脚本名字的字符数改变，这里也要变，否则会导致程序报错 
  resultspath = datasetpath = paste(suppath,"/dataset/",sep="")
  datalist = data_initialize()
  
  
  #看了KSNN的论文，做法是随机分两半，然后测试集做5折交叉验证，因此过采样应该用于脑龄预测而不可用于测试集改变测试集性质
  #算法性能测试
  test_acc_list = method_name_list = dataset_name_list = c()
  if (T) {
    for(i in c(1: (length(datalist)) )){#
      dataset = datalist[[i]]
      datalabel = dataset$datalabel
      
      ranges = unique(datalabel)
      ncluster = length(ranges)
      data = dataset$data
      idnum = dataset$idnum
      idbin = dataset$idbin
      idcat = dataset$idcat
      # 是否归一化对结果影响不大
      # for (j in idnum) {
      #   max_c = max(data[,j])
      #   min_c = min(data[,j])
      #   for (p in c(1:nrow(data))) {
      #     data[p,j] = (data[p,j]-min_c)/(max_c-min_c)
      #   }
      # }
      #还是要分层过采样啊，否则测试用的小数据集结果奇怪，脑龄预测也很不准
      
      folds = list()
      for (j in c(1:10)) {#首先对数据集进行分层随机过采样，保证分折时每层至少能有一个分配到折里（2过采样到10），而且每折里该层的样本数一致（32过采样到40）
        fold = c()
        for (k in ranges) {#分层/分簇
          range_idx = which(datalabel == k)
          get_length = ceiling(length(range_idx)/10)#32个-每折3个-要过采样到40；round(length(range_idx)/10)则是30个就好
          if (get_length < 1) {get_length = 1} #如果这个簇的样本数太少，都做不到每个簇至少1个样本，那就人为规定每个簇1个样本
          unused_idx = setdiff(range_idx, unlist(folds))#放前面而不是后面是为了下面方便用unsed_idx
          if (length(unused_idx) < get_length) {#如果簇内剩下的样本不够抽样填充下面的折了，那就算下这折还差几个，人工随机抽样填满
            #39的第十折，需要抽样补充1个；8的第9和第10折，各补充1个
            set.seed(j)
            fold = c(fold, unused_idx, sample(range_idx, get_length-length(unused_idx)))#不需要设置为可重复，因为肯定够抽一折的量
          } else if (length(unused_idx) == 1) {#加这个是因为R语言特点，如果unused_idx只是一个数值，就当作从1：unused_idx抽样
            fold = c(fold, rep(unused_idx, get_length))
          } else {#最好的情况，比如正好30个，或者39中的前9折
            set.seed(j)
            fold = c(fold, sample(unused_idx, get_length))
          }
        }
        folds[[j]] = fold
      }
      data = data[unlist(folds),]
      datan = data[,idnum]
      datalabel = datalabel[unlist(folds)]
      n = length(datalabel)
      #分层分折
      folds = list()
      for (j in c(1:10)) {
        fold = c()
        for (k in ranges) {
          range_idx = which(datalabel == k)
          get_length = length(range_idx)/10
          unused_idx = setdiff(range_idx, unlist(folds))
          set.seed(j)
          fold = c(fold, sample(unused_idx, get_length))
        }
        folds[[j]] = fold
      }
      folds_length = length(folds[[1]])
      #这种分3个是对比算法性能，也有论文是只分9:1的训练和验证做10次10折取得100个结果/模型然后用于做假设检验配合着MAE一起说明问题，而单纯做一个好模型只要训练+验证集来得到泛化误差和防止模型过拟合https://zhuanlan.zhihu.com/p/31924220
      #难怪留一法泛化误差小，论文里都是随机抽取，都不考虑抽取的的训练、验证、测试集是否具有原数据集的结构/具有原数据集的代表性
      #抽取测试集，留下验证集做多折交叉验证，比例1:1
      set.seed(1) 
      test_idx = c(1:round(n/2)) #离谱，直接是0.5为间隔的，必须加round
      test_dat = data[test_idx,idnum]
      if (length(idnum) == 1) {
        test_dat = as.matrix(test_dat)
      }
      test_datm = data[test_idx,]
      test_lab = datalabel[test_idx]
      #在验证集上做多折交叉验证-随机均分5份数，做5次验证，每次4份训练1份验证，每次都在验证集上网格法调参到最佳，选多次中测试集上最好结果一次的参数进行后续的验证集
      #考虑均分无法整除的情况-前k-1份抽n/(k-1)向上取整，最后一份时用if语句将剩下的给他，结果是最后一次的验证集数量可能会很少，但是符合dbglm要求
      #knn和ksnn为了调参k要在调整集上做交叉验证，svm dbglm 3个半监督为了选择一个最好的模型，做交叉验证，
      #均分k份
      fold_k = 5
      fold_volume = length(datalabel)/10
      rest_idx = setdiff(c(1:n), test_idx)
      rest_dat = data[rest_idx,idnum]
      if (length(idnum) == 1) {
        rest_dat = as.matrix(rest_dat)
      }
      rest_datm = data[rest_idx,]
      rest_lab = datalabel[rest_idx]
      folds = list()
      for (j in c(1:fold_k)) {
        folds[[j]] = rest_idx[(1+folds_length*(j-1)):(folds_length*j)]
      }
      # rest_idx_for_folds = rest_idx
      # for (j in c(1:fold_k)) {
      #   if (j == fold_k) {
      #     folds[[j]] = rest_idx_for_folds
      #   } else {
      #     folds[[j]] = sample(rest_idx_for_folds, size = fold_volume, replace = FALSE)
      #     rest_idx_for_folds = setdiff(rest_idx_for_folds, folds[[j]])
      #   }
      # }
      
      #每份作为验证集，做k次，得最佳结果的参数(我的算法需要改进，对比算法也要挑一些新的；而且我还需要和原算法对比)
      #预测类任务与聚类不同，注意不要整体归一化，会将部分方差信息泄露给训练集
      best_knn = best_ksnn = 
        best_dbglm = best_svm = 
        best_emlsc = best_iclsc = best_mcplda =
        best_smknn = best_smksnn = best_smglm =
        best_tsvm = best_s4vm = best_wsvm = Inf
      # best_knn_k = best_ksnn_k = best_svm_c = best_dbglm_dim = best_dbglm_model = NULL #dbglm似乎不用调参，那就不用做5折交叉验证best_dbglm_model best_dbglm best_smglm
      #可以只算一下我的sm矩阵，然后在对应的knn，ksnn，dbglm那里加一个依赖sm距离矩阵的，这样代码就整洁多了
      for (j in c(1:fold_k)) {
        valid_idx = folds[[j]]
        valid_dat = data[valid_idx,idnum]
        if (length(idnum) == 1) {
          valid_dat = as.matrix(valid_dat)
        }
        valid_datm = data[valid_idx,]
        valid_lab = datalabel[valid_idx]
        
        train_idx = setdiff(as.vector(unlist(folds)), valid_idx)
        train_dat = data[train_idx,idnum]
        if (length(idnum) == 1) {
          train_dat = as.matrix(train_dat)
        }
        train_datm = data[train_idx,]
        train_lab = datalabel[train_idx]
        
        
        #knn函数用法参考 R中的knn算法实现 https://zhuanlan.zhihu.com/p/31399056#:~:text=2.R%E5%8C%85%E4%B8%AD%E7%9A%84knn%E5%AE%9E%E7%8E%B0%201%202.1%20class%E5%8C%85%E4%B8%AD%E7%9A%84knn%20knn%20%28%29%E5%87%BD%E6%95%B0%E7%9A%84%E8%AF%AD%E6%B3%95%E5%92%8C%E5%8F%82%E6%95%B0%E5%A6%82%E4%B8%8B%EF%BC%9A%20knn%20%28train%2C,%E5%B8%83%E5%B0%94%E5%80%BC%EF%BC%8C%E6%8C%87%E7%A4%BA%E6%98%AF%E5%90%A6%E5%9C%A8KNN%E9%A2%84%E6%B5%8B%E5%89%8D%E5%B0%86%E6%95%B0%E6%8D%AE%E6%A0%87%E5%87%86%E5%8C%96%EF%BC%8C%E9%BB%98%E8%AE%A4%E4%B8%BATRUE%20...%203%202.3%20kknn%E5%8C%85%E4%B8%AD%E7%9A%84kknn%20kknn%20%28%29%E5%87%BD%E6%95%B0%E7%9A%84%E8%AF%AD%E6%B3%95%E5%92%8C%E5%8F%82%E6%95%B0%E5%A6%82%E4%B8%8B%EF%BC%9A%20
        #注意knn只能利用数值变量
        for (k in c(1:(length(train_lab)))) {
          knn_result = knn(train_dat, valid_dat, train_lab, k = k)
          knn_result = sum(abs(as.numeric(knn_result) - valid_lab))#还需要将knn结果转化为数值向量才能做减法
          if (knn_result < best_knn) {#这里的=可能对结果产生影响，这里是倾向于选较大的K值
            best_knn = knn_result
            best_knn_k = k
          }
        }
        #ksnn 一次竟然只能预测一个对象，标签只是训练集，距离矩阵是训练集和验证集的一个，而且为了避免错误，按照示例来做
        distdata = as.matrix(dist(datan, method = "euclidean"))#注意ksnn只能计算数值变量
        pred_ksNN = rep(0, length(valid_idx))
        for (k in c(1:length(train_idx))) {
          for (p in c(1:length(valid_idx))) {
            ksnn_label = train_lab
            ksnn_distance = distdata[valid_idx[p],train_idx] #根据ksnn示例，距离里不包括自己到自己的距离,而且只包括自己到训练集的距离
            pred_ksNN[p] = rcpp_ksNN(ksnn_label, ksnn_distance, k)$pred
            # pred_ksNN[p] = round(as.numeric(pred_ksNN[p]))
          }
          ksnn_result = sum(abs(pred_ksNN - valid_lab))
          if (ksnn_result < best_ksnn) {
            best_ksnn = ksnn_result
            best_ksnn_k = k
          }
        }
        
        #smksnn
        for (l in c(2:length(train_lab))) {
          pred_ksNN = rep(0, length(valid_idx))
          #这里得到的距离矩阵是个rest集的距离矩阵，需要映射回全集，方便下面计算
          rest_distdata = smdistance(rest_datm, rest_lab, rest_idx, train_datm, train_lab, train_idx, idnum, idbin, idcat, l)
          distdata = matrix(0,length(datalabel),length(datalabel))
          for (k in c(1:length(rest_idx))) {
            for (p in c(1:length(rest_idx))) {
              distdata[rest_idx[k],rest_idx[p]] = rest_distdata[k,p]
            }
          }
          
          for (k in c(1:length(train_idx))) {
            for (p in c(1:length(valid_idx))) {
              ksnn_label = train_lab
              ksnn_distance = distdata[valid_idx[p],train_idx] #根据ksnn示例，距离里不包括自己到自己的距离,而且只包括自己到训练集的距离
              pred_ksNN[p] = rcpp_ksNN(ksnn_label, ksnn_distance, k)$pred
              # pred_ksNN[p] = round(as.numeric(pred_ksNN[p]))
            }
            smksnn_result = sum(abs(pred_ksNN - valid_lab))
            if (smksnn_result < best_smksnn) {
              best_smksnn = smksnn_result
              best_smksnn_k = k
              best_minpts = l
            }
          }
        }
        
        

        # tem_train_datm = train_datm
        # tem_train_lab = train_lab
        # tem_train_idx = train_idx
        # tem_valid_idx = 
        # 
        # pred_ksNN = rep(0, length(valid_idx))
        # #这里得到的距离矩阵是个rest集的距离矩阵，需要映射回全集，方便下面计算
        # rest_distdata = smdistance(rest_datm, rest_lab, rest_idx, tem_train_datm, tem_train_lab, tem_train_idx, idnum, idbin, idcat)
        # distdata = matrix(0,length(datalabel),length(datalabel))
        # for (k in c(1:length(rest_idx))) {
        #   for (p in c(1:length(rest_idx))) {
        #     distdata[rest_idx[k],rest_idx[p]] = rest_distdata[k,p]
        #   }
        # }
        # 
        # for (p in c(1:length(valid_idx))) {
        #   ksnn_label = train_lab
        #   ksnn_distance = distdata[p,train_idx] #根据ksnn示例，距离里不包括自己到自己的距离,而且只包括自己到训练集的距离
        #   pred_ksNN[p] = rcpp_ksNN(ksnn_label, ksnn_distance, k)$pred
        #   # pred_ksNN[p] = round(as.numeric(pred_ksNN[p]))
        # }
        # smksnn_result = sum(abs(pred_ksNN - valid_lab))
        
        
        
        
        # for (k in c(2:(length(rest_lab)/ncluster))) {
        #   smksnn_result = smskm(train_datm, train_lab, valid_datm, idnum, idbin, idcat)
        #   smksnn_result = sum(abs(smksnn_result - valid_lab))
        #   if (smksnn_result < best_smksnn) {
        #     best_smksnn = smksnn_result
        #     best_minpts = k
        #   }
        # }
        
        # rest_distdata = smdistance(rest_datm, rest_lab, rest_idx, train_datm, train_lab, train_idx, idnum, idbin, idcat)
        # distdata = matrix(0,length(datalabel),length(datalabel))
        # for (k in c(1:length(rest_idx))) {
        #   for (p in c(1:length(rest_idx))) {
        #     distdata[rest_idx[k],rest_idx[p]] = rest_distdata[k,p]
        #   }
        # }
        # smdatalabel = datalabel
        # smdatalabel[-train_idx] = NA
        # 
        # for (k in c(1:length(train_idx))) {
        #   smksnn_result = distknn(distdata, smdatalabel, train_idx, valid_idx, k)
        #   smksnn_result = sum(abs(smksnn_result[valid_idx] - valid_lab))
        #   if (smksnn_result < best_smksnn) {
        #     best_smksnn = smksnn_result
        #     best_smksnn_k = k
        #   }
        # }
        
        
        
        # #dbglm
        # D2 = as.matrix(dist(data[train_idx,idnum], method = "euclidean"))^2
        # class(D2) = "D2"
        # dbglm_model = dbglm(D2, train_lab, family = poisson(link = "log"), method="rel.gvar")
        # 
        # sup_num = length(train_idx)-length(valid_idx)
        # set.seed(1)
        # sup_idx = sample(train_idx, sup_num, replace = F)
        # 
        # D2 = as.matrix(dist(data[c(valid_idx,sup_idx),idnum], method = "euclidean"))^2
        # class(D2) = "D2"
        # dbglm_result = predict(dbglm_model, D2, type.pred="response",type.var="D2")[1:length(valid_idx)]
        # dbglm_result = sum(abs(dbglm_result - valid_lab))
        # if (dbglm_result < best_dbglm) {
        #   best_dbglm = dbglm_result
        #   best_dbglm_model = dbglm_model
        #   best_dbglm_dim = length(train_idx)
        # }
        
        #smglm
        for (l in c(2:length(train_lab))) {
          rest_distdata = smdistance(rest_datm, rest_lab, rest_idx, train_datm, train_lab, train_idx, idnum, idbin, idcat, l)
          rest_distdata = rest_distdata^2
          distdata = matrix(0,length(datalabel),length(datalabel))
          for (k in c(1:length(rest_idx))) {
            for (p in c(1:length(rest_idx))) {
              distdata[rest_idx[k],rest_idx[p]] = rest_distdata[k,p]
            }
          }
          train_distdata = distdata[train_idx,train_idx]
          class(train_distdata) = "D2"
          smglm_model = dbglm(train_distdata, train_lab, family = poisson(link = "log"), method="rel.gvar")
          
          sup_num = length(train_idx)-length(valid_idx)
          set.seed(1)
          sup_idx = sample(train_idx, sup_num, replace = F)
          
          valid_distdata = distdata[c(valid_idx,sup_idx),c(valid_idx,sup_idx)]
          class(valid_distdata) = "D2"
          smglm_result = predict(smglm_model, valid_distdata, type.pred="response",type.var="D2")[1:length(valid_idx)]
          smglm_result = sum(abs(smglm_result - valid_lab))
          if (smglm_result < best_smglm) {
            best_smglm = smglm_result
            best_smglm_minpts = l
          }
        }
        
        #lsvm 文献里搜索范围是2^-7到2^7，这里只是跑一跑作为一个参考 tsvm s4vm wellsvm,我都只调节正则化参数/惩罚系数C（0-无穷大），而且在无标签数据上的c应该比有标签数据上的c小以表示其惩罚能力低
        for (k in c(-8:8)) {
          svm_model = svm(train_dat, train_lab, cost  = 2^k, kernel = "linear",scale = F)
          svm_result = predict(svm_model, valid_dat)
          svm_result = sum(abs(svm_result - valid_lab))
          if (svm_result < best_svm) {#这里的=可能对结果产生影响，这里是倾向于选较大的K值
            best_svm = svm_result
            best_svm_c = k
          }

          if (i>0 && i<6) {
            #tsvm s4vm wellsvm 在实际数据集上报错'labels'不对；长度3应该是一或2，因为它只能解决二元分类问题
            # tsvm_model = TSVM(X = train_dat, y = as.factor(train_lab), X_u = valid_dat, 2^k, 2^k/10)#只要把标签变成factor向量就行了
            # tsvm_result = RSSL::predict(tsvm_model, valid_dat)
            # tsvm_result = sum(abs(as.numeric(tsvm_result) - valid_lab))#这里变回数值向量，否则无法计算
            # if (is.na(tsvm_result)==F && tsvm_result < best_tsvm) {#这里的=可能对结果产生影响，这里是倾向于选较大的K值
            #   best_tsvm = tsvm_result
            #   best_tsvm_c = k
            # }
            # s4vm_model = S4VM(train_dat, as.factor(train_lab), valid_dat, 2^k, 2^k/100)#只要把标签变成factor向量就行了
            # s4vm_result = predict(s4vm_model, valid_dat[,idnum])
            # s4vm_result = sum(abs(as.numeric(s4vm_result) - valid_lab))
            # if (s4vm_result < best_s4vm) {#这里的=可能对结果产生影响，这里是倾向于选较大的K值
            #   best_s4vm = s4vm_result
            #   best_s4vm_c = k
            # }
            wsvm_model = WellSVM(train_dat, as.factor(train_lab), valid_dat, 2^k, 2^k/10)#只要把标签变成factor向量就行了
            wsvm_result = predict(wsvm_model, valid_dat)
            wsvm_result = sum(abs(as.numeric(wsvm_result) - valid_lab))
            if (wsvm_result < best_wsvm) {#这里的=可能对结果产生影响，这里是倾向于选较大的K值
              best_wsvm = wsvm_result
              best_wsvm_c = k
            }
          }

        }
        
        # #emlsc iclsc mcplda
        # em_lab = as.factor(datalabel) #将数值向量转化为因子向量是能够使用算法的关键一步
        # em_dat = cbind(data[,idnum], em_lab)
        # em_dat[valid_idx, which(colnames(em_dat)=="em_lab")] = NA #获取标签的列名的序号位，并且将对应位置设置为NA
        # em_dat = em_dat[rest_idx,]
        # 
        # emlsc_model = EMLeastSquaresClassifier(em_lab~.,em_dat)
        # emlsc_result = predict(emlsc_model,valid_dat)
        # emlsc_result = sum(abs(as.numeric(emlsc_result) - valid_lab))#结果是因子向量，要转化回数值向量才能运算
        # if (emlsc_result < best_emlsc) {#这里的=可能对结果产生影响，这里是倾向于选较大的K值
        #   best_emlsc = emlsc_result
        #   best_emlsc_model = emlsc_model
        # }
        # iclsc_model = ICLeastSquaresClassifier(em_lab~.,em_dat, projection = "semisupervised")
        # iclsc_result = predict(iclsc_model,valid_dat)
        # iclsc_result = sum(abs(as.numeric(iclsc_result) - valid_lab))#结果是因子向量，要转化回数值向量才能运算
        # if (iclsc_result < best_iclsc) {#这里的=可能对结果产生影响，这里是倾向于选较大的K值
        #   best_iclsc = iclsc_result
        #   best_iclsc_model = iclsc_model
        # }
        # mcplda_model = MCPLDA(train_dat[,idnum], as.factor(train_lab), valid_dat[,idnum])
        # mcplda_result = predict(mcplda_model,valid_dat)
        # mcplda_result = sum(abs(as.numeric(mcplda_result) - valid_lab))#结果是因子向量，要转化回数值向量才能运算
        # if (mcplda_result < best_mcplda) {#这里的=可能对结果产生影响，这里是倾向于选较大的K值
        #   best_mcplda = mcplda_result
        #   best_mcplda_model = mcplda_model
        # }
        
      cat("dataset",i,"fold",j,".\n")
      }
      
      #选参完毕，开始测试
      if(T){
        #knn
        knn_result = knn(rest_dat, test_dat, rest_lab, k = best_knn_k)
        knn_result = sum(abs(as.numeric(knn_result) - test_lab))#还需要将knn结果转化为数值向量才能做减法
        
        #ksnn 一次竟然只能预测一个对象，标签只是训练集，距离矩阵是训练集和验证集的一个，而且为了避免错误，按照示例来做
        distdata = as.matrix(dist(datan, method = "euclidean"))#注意ksnn只能计算数值变量
        pred_ksNN = rep(0, length(test_idx))
        for (p in c(1:length(test_idx))) {
          ksnn_label = rest_lab
          ksnn_distance = distdata[test_idx[p],rest_idx] #根据ksnn示例，距离里不包括自己到自己的距离,而且只包括自己到训练集的距离
          pred_ksNN[p] = rcpp_ksNN(ksnn_label, ksnn_distance, best_ksnn_k)$pred
        }
        ksnn_result = sum(abs(pred_ksNN - test_lab))
        
        #smksnn
        distdata = smdistance(data, datalabel, c(1:length(datalabel)),rest_datm, rest_lab, rest_idx, idnum, idbin, idcat, best_minpts)
        pred_ksNN = rep(0, length(test_idx))
        for (p in c(1:length(test_idx))) {
          ksnn_label = rest_lab
          ksnn_distance = distdata[test_idx[p],rest_idx] #根据ksnn示例，距离里不包括自己到自己的距离,而且只包括自己到训练集的距离
          pred_ksNN[p] = rcpp_ksNN(ksnn_label, ksnn_distance, best_smksnn_k)$pred
        }
        smksnn_result = sum(abs(pred_ksNN - test_lab))
        
        
        # distdata = smdistance(data, datalabel, c(1:length(datalabel)),rest_datm, rest_lab, rest_idx, idnum, idbin, idcat)
        # 
        # smdatalabel = datalabel
        # smdatalabel[-train_idx] = NA
        # 
        # smksnn_result = distknn(distdata, smdatalabel, train_idx, test_idx, best_smksnn_k)
        # smksnn_result = sum(abs(smksnn_result[test_idx] - test_lab))
        
       
        
        
        # smksnn_result = smskm(train_datm, train_lab, test_datm, idnum, idbin, idcat)
        # smksnn_result = sum(abs(smksnn_result - test_lab))
        
        #lsvm 
        svm_model = svm(rest_dat, rest_lab, cost  = 2^best_svm_c, kernel = "linear",scale = F)
        svm_result = predict(svm_model, test_dat)
        svm_result = sum(abs(svm_result - test_lab))
        
        if (i>0 && i<6) {
          # tsvm_model = TSVM(rest_dat, as.factor(rest_lab), test_dat, 2^best_tsvm_c, 2^best_tsvm_c/10)#只要把标签变成factor向量就行了
          # tsvm_result = predict(tsvm_model, test_dat)
          # tsvm_result = sum(abs(as.numeric(tsvm_result) - test_lab))

          # s4vm_model = S4VM(rest_dat, as.factor(rest_lab), test_dat, 2^best_s4vm_c, 2^best_s4vm_c/10)#只要把标签变成factor向量就行了
          # s4vm_result = predict(s4vm_model, test_dat)
          # s4vm_result = sum(abs(as.numeric(s4vm_result) - test_lab))

          wsvm_model = WellSVM(rest_dat, as.factor(rest_lab), test_dat, 2^best_wsvm_c, 2^best_wsvm_c/10)#只要把标签变成factor向量就行了
          wsvm_result = predict(wsvm_model, test_dat)
          wsvm_result = sum(abs(as.numeric(wsvm_result) - test_lab))
        }
        
        #emlsc iclsc mcplda 前两者只能用于二分类任务，所以放弃，而且这下也不用考虑这部分的调参问题了
        # em_lab = as.factor(datalabel) #将数值向量转化为因子向量是能够使用算法的关键一步
        # em_dat = cbind(data[,idnum], em_lab)
        # em_dat[test_idx, which(colnames(em_dat)=="em_lab")] = NA #获取标签的列名的序号位，并且将对应位置设置为NA
        # 

        # emlsc_model = EMLeastSquaresClassifier(em_lab~.,em_dat)
        # emlsc_result = predict(emlsc_model,test_dat)
        # emlsc_result = sum(abs(as.numeric(emlsc_result) - test_lab))#结果是因子向量，要转化回数值向量才能运算
        # 
        # iclsc_model = ICLeastSquaresClassifier(em_lab~.,em_dat, projection = "semisupervised")
        # iclsc_result = predict(iclsc_model,test_dat)
        # iclsc_result = sum(abs(as.numeric(iclsc_result) - test_lab))#结果是因子向量，要转化回数值向量才能运算

        mcplda_model = MCPLDA(rest_dat, as.factor(rest_lab), test_dat)
        mcplda_result = predict(mcplda_model,test_dat)
        mcplda_result = sum(abs(as.numeric(mcplda_result) - test_lab))#结果是因子向量，要转化回数值向量才能运算
        
        
        #dbglm
        distdata = as.matrix(dist(datan, method = "euclidean"))^2#不要直接出rest_idx，因为后面测试距离矩阵也要用这个抽取出来
        rest_distdata = distdata[rest_idx,rest_idx]#不能再dbglm里做这一步，因为类型变为D2了
        class(rest_distdata) = "D2"
        dbglm_model = dbglm(rest_distdata, rest_lab, family = poisson(link = "log"), method="rel.gvar")

        sup_num = length(rest_idx)-length(test_idx)
        set.seed(1)
        sup_idx = sample(rest_idx, sup_num, replace = F)

        test_distdata = distdata[c(rest_idx,sup_idx), c(rest_idx,sup_idx)]
        class(test_distdata) = "D2"
        dbglm_result = predict(dbglm_model, test_distdata, type.pred="response",type.var="D2")[1:length(valid_idx)]
        dbglm_result = sum(abs(dbglm_result - test_lab))
        
        #smglm
          distdata = smdistance(data, datalabel, c(1:length(datalabel)), rest_datm, rest_lab, rest_idx, idnum, idbin, idcat, best_smglm_minpts)
          distdata = distdata^2
          rest_distdata = distdata[rest_idx,rest_idx]
          class(rest_distdata) = "D2"
          smglm_model = dbglm(rest_distdata, rest_lab, family = poisson(link = "log"), method="rel.gvar")
          
          # sup_num = length(datalabel)-length(rest_lab)
          # set.seed(1)
          # sup_idx = sample(rest_idx, sup_num, replace = F)
          
          test_distdata = distdata[c(test_idx),c(test_idx)]#因为过采样和对半分，所以一样多，不用抽补
          class(test_distdata) = "D2"
          smglm_result = predict(smglm_model, test_distdata, type.pred="response",type.var="D2")[1:length(valid_idx)]
          smglm_result = sum(abs(smglm_result - valid_lab))
          
        
      }
      
      if (i>0 && i<6) {
        test_acc_list = c(test_acc_list, 
                          knn_result/length(datalabel), 
                          ksnn_result/length(datalabel), 
                          smksnn_result/length(datalabel),
                          dbglm_result/length(datalabel), 
                          smglm_result/length(datalabel), 
                          svm_result/length(datalabel),
                          # tsvm_result/length(datalabel),
                          # s4vm_result/length(datalabel),
                          wsvm_result/length(datalabel),
                          # emlsc_result/length(datalabel),
                          # iclsc_result/length(datalabel),
                          mcplda_result/length(datalabel)
        )
        method_name_list = c(method_name_list, 
                             "KNN", "KSNN", "SMKSNN","DBGLM","SMGLM", "SVM","WSVM", "MCPLDA")#
        dataset_name_list = c(dataset_name_list , rep(names(datalist[i]),8) )
      } else {
        test_acc_list = c(test_acc_list, 
                          knn_result/length(datalabel), 
                          ksnn_result/length(datalabel), 
                          smksnn_result/length(datalabel),
                          dbglm_result/length(datalabel), 
                          smglm_result/length(datalabel), 
                          svm_result/length(datalabel),
                          mcplda_result/length(datalabel)
        )
        method_name_list = c(method_name_list, 
                             "KNN", "KSNN", "SMKSNN","DBGLM","SMGLM", "SVM", "MCPLDA")#
        dataset_name_list = c(dataset_name_list , rep(names(datalist[i]),7) )
      }
      
    }
  }
  
  out = data.frame(
    result_mae = test_acc_list,
    method_name = method_name_list,
    dataset_name = dataset_name_list
  )
  save(out, file = paste(resultspath, "test_result.RData", sep=""))
  
  #改进算法和原算法对比
  out_new = out[(out$method_name=="KNN" | out$method_name=="KSNN" | out$method_name=="SMKSNN") ,]#给前面加括号是必要的#& out$dataset_name!="SALD"
  print(ggplot(data=out_new, mapping=aes(x = dataset_name, y = result_mae,fill=method_name)) 
        + geom_bar(stat="identity",position=position_dodge(0.75),width=0.6) 
        + labs(title = NULL, x ="Datasets", y = "MAE") 
        + theme_bw() 
        + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5, angle = 20),axis.text.y = element_text(vjust = 0.5, hjust = 0.5))
        + theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank())
        + scale_fill_manual(values =c("#16048A", "#6201A9", "#9E189F", "#CC4A76", "#EA7854", "#FDB330","#f8e620"))
        + theme(axis.title.y = element_text(angle = 0, vjust = 0.5)))
  ggsave(paste(datasetpath,"comparedwith origin SKNN.pdf", sep=""), device = cairo_pdf,width =5, height =3.75)
  
  out_new = out[(out$method_name=="DBGLM" | out$method_name=="SMGLM") ,]#给前面加括号是必要的#& out$dataset_name!="SALD"
  print(ggplot(data=out_new, mapping=aes(x = dataset_name, y = result_mae,fill=method_name)) 
        + geom_bar(stat="identity",position=position_dodge(0.75),width=0.6) 
        + labs(title = NULL, x ="Datasets", y = "MAE") 
        + theme_bw() 
        + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5, angle = 20),axis.text.y = element_text(vjust = 0.5, hjust = 0.5))
        + theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank())
        + scale_fill_manual(values =c("#16048A", "#6201A9", "#9E189F", "#CC4A76", "#EA7854", "#FDB330","#f8e620"))
        + theme(axis.title.y = element_text(angle = 0, vjust = 0.5)))
  ggsave(paste(datasetpath,"comparedwith origin DBGLM.pdf", sep=""), device = cairo_pdf,width =5, height =3.75)
  #改进算法和MCPLDA算法对比
  out_new = out[(out$method_name=="SMKSNN" | out$method_name=="MCPLDA" | out$method_name=="SMGLM") ,]#& out$dataset_name!="SALD"
  print(ggplot(data=out_new, mapping=aes(x = dataset_name, y = result_mae,fill=method_name)) 
        + geom_bar(stat="identity",position=position_dodge(0.75),width=0.6) 
        + labs(title = NULL, x ="Datasets", y = "MAE") 
        + theme_bw() 
        + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5, angle = 20),axis.text.y = element_text(vjust = 0.5, hjust = 0.5))
        + theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank())
        + scale_fill_manual(values =c("#16048A", "#6201A9", "#9E189F", "#CC4A76", "#EA7854", "#FDB330","#f8e620"))
        + theme(axis.title.y = element_text(angle = 0, vjust = 0.5)))
  ggsave(paste(datasetpath,"comparedwith MCPLDA.pdf", sep=""),device = cairo_pdf,width =5, height =3.75)
  #改进算法和wsvm半监督算法在二分类人工数据集上对比
  out_new = out[(out$method_name=="SMKSNN" | out$method_name=="SVM" | out$method_name=="WSVM" | out$method_name=="SMGLM") & (out$dataset_name == "NOISE" | out$dataset_name == "RING" | out$dataset_name == "HALFRING" | out$dataset_name == "OVERLAPPING" | out$dataset_name == "FERTILITY"),]
  print(ggplot(data=out_new, mapping=aes(x = dataset_name, y = result_mae,fill=method_name)) 
        + geom_bar(stat="identity",position=position_dodge(0.75),width=0.6) 
        + labs(title = NULL, x ="Datasets", y = "MAE") 
        + theme_bw() 
        + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5, angle = 20),axis.text.y = element_text(vjust = 0.5, hjust = 0.5))
        + theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank())
        + scale_fill_manual(values =c("#16048A", "#6201A9", "#9E189F", "#CC4A76", "#EA7854", "#FDB330","#f8e620"))
        + theme(axis.title.y = element_text(angle = 0, vjust = 0.5)))
  ggsave(paste(datasetpath,"comparedwith svm.pdf", sep=""),device = cairo_pdf,width =5, height =3.75)
  return(out)
}
smskm_no = function(train_dat, train_lab, test_dat, idnum, idbin, idcat){
  
  
  ncluster = length(unique(train_lab))
  
  test_lab = c()
  for (j in c(1:nrow(test_dat))) {
    
    #生成训练集约束
    mustLink = NULL
    cantLink = NULL
    allink = t(combn(c(1:length(train_lab)),2))
    for(i in c(1:nrow(allink))){
      if(train_lab[allink[i,1]] == train_lab[allink[i,2]]){mustLink = rbind(mustLink, allink[i,])}
      if(train_lab[allink[i,1]] != train_lab[allink[i,2]]){cantLink = rbind(cantLink, allink[i,])}
    }
    #训练集和测试集的第i个合并为新训练集sub_dat，一个个地预测比总体预测准确性高不一定，但是判断结果簇号到底是哪个更不容易偏差
    sub_dat = rbind(train_dat,test_dat[j,])#预测对象加后面，从而与ml和cl的序号兼容
    
    #计算距离矩阵
    if(is.null(idbin) && is.null(idcat)){ 
      sub_distdata = as.matrix(dist(sub_dat, method = "euclidean"))
    } else {
      sub_distdata = distmix(sub_dat, method = "gower", idnum = idnum, idbin = idbin, idcat = idcat)
    }
    #施加约束
    if(length(mustLink)>0){
      for(i in c(1:nrow(mustLink))){
        sub_distdata[mustLink[i,2], mustLink[i,1]] = sub_distdata[mustLink[i,1], mustLink[i,2]] = 0.5*sub_distdata[mustLink[i,1], mustLink[i,2]]
      }
    }
    if(length(cantLink)>0){
      for(i in c(1:nrow(cantLink))){
        sub_distdata[cantLink[i,2], cantLink[i,1]] = sub_distdata[cantLink[i,1], cantLink[i,2]] = 2*sub_distdata[cantLink[i,1], cantLink[i,2]]
      }
    }
    #半监督聚类
    skm_label = skm(sub_distdata, ncluster)$cluster
    #结果核定
    test_point_lab = skm_label[length(train_lab)+1]
    test_point_sits = which(skm_label == test_point_lab)
    may_lab = train_lab[test_point_sits[-length(test_point_sits)]]
    mode_result = table(may_lab)
    test_point_lab_true = as.numeric(names(mode_result[which.max(mode_result)]))#求众数
    test_lab = c(test_lab,test_point_lab_true)
    # #结果加入训练集
    # train_dat = rbind(train_dat, test_dat[j,])
    # train_lab = c(train_lab, test_point_lab_true)
  }
  
  
  # return(test_lab)
  return(test_lab)
}
smskm = function(train_dat, train_lab, test_dat, idnum, idbin, idcat){
  
  
  ncluster = length(unique(train_lab))
  rounds = nrow(test_dat)
  
  test_lab = c()
  for (k in c(1:rounds)) {
    
    #生成训练集约束
    mustLink = NULL
    cantLink = NULL
    allink = t(combn(c(1:length(train_lab)),2))
    for(i in c(1:nrow(allink))){
      if(train_lab[allink[i,1]] == train_lab[allink[i,2]]){mustLink = rbind(mustLink, allink[i,])}
      if(train_lab[allink[i,1]] != train_lab[allink[i,2]]){cantLink = rbind(cantLink, allink[i,])}
    }
    
    max_sil = -10
    for (j in c(1:nrow(test_dat))) {
      #训练集和测试集的第i个合并为新训练集sub_dat，一个个地预测比总体预测准确性高不一定，但是判断结果簇号到底是哪个更不容易偏差
      sub_dat = rbind(train_dat,test_dat[j,])#预测对象加后面，从而与ml和cl的序号兼容
      
      #计算距离矩阵
      if(is.null(idbin) && is.null(idcat)){ 
        sub_distdata = as.matrix(dist(sub_dat, method = "euclidean"))
      } else {
        sub_distdata = distmix(sub_dat, method = "gower", idnum = idnum, idbin = idbin, idcat = idcat)
      }
      #施加约束
      if(length(mustLink)>0){
        for(i in c(1:nrow(mustLink))){
          sub_distdata[mustLink[i,2], mustLink[i,1]] = sub_distdata[mustLink[i,1], mustLink[i,2]] = 0.5*sub_distdata[mustLink[i,1], mustLink[i,2]]
        }
      }
      if(length(cantLink)>0){
        for(i in c(1:nrow(cantLink))){
          sub_distdata[cantLink[i,2], cantLink[i,1]] = sub_distdata[cantLink[i,1], cantLink[i,2]] = 2*sub_distdata[cantLink[i,1], cantLink[i,2]]
        }
      }
      
      #半监督聚类
      skm_label = skm(sub_distdata, ncluster)$cluster
      #结果核定
      test_point_lab = skm_label[length(train_lab)+1]
      test_point_sits = which(skm_label == test_point_lab)
      may_lab = train_lab[test_point_sits[-length(test_point_sits)]]
      mode_result = table(may_lab)
      test_point_lab_true = as.numeric(names(mode_result[which.max(mode_result)]))#求众数
      
      sil = mean(silhouette(skm_label, sub_distdata))
      #结果加入训练集引起的sil改变
      # dis_sil = new_sil-old_sil
      
     
      if (sil > max_sil) {
        max_sil = sil
        best_add_dat = j
      }
      
    }
    train_dat = rbind(train_dat, test_dat[best_add_dat,])
    train_lab = c(train_lab, test_point_lab_true)
    test_dat = test_dat[-best_add_dat,]
    test_lab = c(test_lab,test_point_lab_true)
  }
    
  
  
  # return(test_lab)
  return(test_lab)
}
smdistance = function(data, datalabel, dat_idx, sub_dat, sub_lab, sub_idx, idnum, idbin, idcat, l){
  #必须在子集水平上做，因为gower那里的归一化的影响，但是注意对于训练+验证是两者之和的rest集放进，对于rest+测试是全集放进来
  #用训练集的标签产生约束
  
  if (TRUE) {
    mustLink = NULL
    cantLink = NULL
    allink = t(combn(c(1:length(sub_lab)),2))
    for(i in c(1:nrow(allink))){
      if(sub_lab[allink[i,1]] == sub_lab[allink[i,2]]){mustLink = rbind(mustLink, allink[i,])}
      if(sub_lab[allink[i,1]] != sub_lab[allink[i,2]]){cantLink = rbind(cantLink, allink[i,])}
    }
  }
  #训练集的混合变量距离矩阵并加权
  if(is.null(idbin) && is.null(idcat)){ 
    sub_distdata = as.matrix(dist(sub_dat, method = "euclidean"))
  } else {
    sub_distdata = distmix(sub_dat, method = "gower", idnum = idnum, idbin = idbin, idcat = idcat)
  }
  m = nrow(mustLink)
  c = nrow(cantLink)
  if(m>0){
    for(i in c(1:m)){
      sub_distdata[mustLink[i,2], mustLink[i,1]] = sub_distdata[mustLink[i,1], mustLink[i,2]] = 0.5*sub_distdata[mustLink[i,1], mustLink[i,2]]
    }
  }
  if(c>0){
    for(i in c(1:c)){
      sub_distdata[cantLink[i,2], cantLink[i,1]] = sub_distdata[cantLink[i,1], cantLink[i,2]] = 2*sub_distdata[cantLink[i,1], cantLink[i,2]]
    }
  }
  #将训练集距离矩阵映射回全集的混合变量距离矩阵,但是对于训练/验证集，序号表示全集顺序而不是rest集顺序，因此要将训练集序号映射为其在rest集中的序号
  if(is.null(idbin) && is.null(idcat)){ 
    distdata = as.matrix(dist(data, method = "euclidean"))
  } else {
    distdata = distmix(data, method = "gower", idnum = idnum, idbin = idbin, idcat = idcat)
  }
  
  for (i in c(1:length(sub_idx))) {
    sub_idx[i] = which(dat_idx==sub_idx[i])
  }
  
  for (i in c(1:length(sub_idx))) {
    for (j in c(1:length(sub_idx))) {
      distdata[sub_idx[i],sub_idx[j]] = sub_distdata[i,j]
    }
  }
  #用全集的混合变量加权距离矩阵-计算聚类先验矩阵，从而将约束数通过聚类进行泛化和放大
  if (T) {
    ncluster = length(unique(sub_lab)) #这可能提高训练集效果但可能降低验证测试集效果
    #skm_label = skm(distdata, ncluster)$cluster
    skm_label = hdbscan(distdata, l)$cluster
    mustLink = NULL
    cantLink = NULL
    allink = t(combn(c(1:length(skm_label)),2))
    for(i in c(1:nrow(allink))){
      if(skm_label[allink[i,1]] == skm_label[allink[i,2]]){mustLink = rbind(mustLink, allink[i,])}
      if(skm_label[allink[i,1]] != skm_label[allink[i,2]]){cantLink = rbind(cantLink, allink[i,])}
    }
    m = nrow(mustLink)
    c = nrow(cantLink)
    
    #这里应该用一个没有约束加权的还是用加权的混合变量距离矩阵来进一步加权如下？
    if(length(m)!=0){
      for(i in c(1:m)){
        distdata[mustLink[i,2], mustLink[i,1]] = distdata[mustLink[i,1], mustLink[i,2]] = distdata[mustLink[i,1], mustLink[i,2]]*0.5
      }
    }
    if(length(c)!=0){
      for(i in c(1:c)){
        distdata[cantLink[i,2], cantLink[i,1]] = distdata[cantLink[i,1], cantLink[i,2]] = distdata[cantLink[i,1], cantLink[i,2]]*2
      }
    }
  }
  return(distdata)
}

data_initialize = function(){
  scriptpath = rstudioapi::getSourceEditorContext()$path
  scriptpathn = nchar(scriptpath)
  suppath = substr(scriptpath, 1, scriptpathn-8)#注意这个脚本名字的字符数改变，这里也要变，否则会导致程序报错 
  datasetpath = paste(suppath,"/dataset/",sep="")
  #生成人工数据集
  if (T) {
    # Generate data set with normal distribution and overlap
    if (TRUE) {
      set.seed(1)
      # x1 = x2 = y1 = y2 = c()
      # for (i in c(1:10)) {
      #   x1 = c(x1,rep(i,10))
      #   y1 = c(y1,c(1:10))
      #   x2 = c(x2,rep(i+2.5,10))
      #   y2 = c(y2,c(3.5:12.5))
      # }
      
      # x1 = rnorm(50, mean = 60, sd = 1)
      # y1 = rnorm(50, mean = 60, sd = 1)
      x1 = round(runif(50, 200, 700))
      y1 = round(runif(50, 200, 700))
      # # x = sample(c(200:700),100)
      # # y = sample(c(200:700),100)
      # # x1 = sample(x,50)
      # # y1 = sample(y,50)
      # # x2 = intersect(x,x1)
      # # y2 = intersect(y,y1)
      
      # x2 = rnorm(50, mean = 60, sd = 1)
      # y2 = rnorm(50, mean = 60, sd = 1)
      x2 = round(runif(50, 200, 700))+0.5
      y2 = round(runif(50, 200, 700))+0.5
      z1 = rep(100,length(x1))
      z2 = rep(1,length(x2))
      cluster1 = data.frame(x = x1, y = y1, z = z1, label = "cluster 1")
      cluster2 = data.frame(x = x2, y = y2, z = z2, label = "cluster 2")
      overlapping_clusters <- rbind(cluster1, cluster2)
      #search overlap points
      equalidx = c()
      distdata = as.matrix(dist(overlapping_clusters[,c(1:2)]))
      equalist = which(distdata==0, arr.ind = TRUE)
      for(i in c(1:nrow(equalist))){
        if(equalist[i,1] != equalist[i,2]){
          equalidx = c(equalidx, equalist[i,1], equalist[i,2])
        }
      }
      equalidx
      save(overlapping_clusters, file=paste(datasetpath, "overlapping_clusters.RData", sep=""))
      
      #2D plot
      # print(ggplot(data = overlapping_clusters, mapping = aes(x = x, y = y, shape = label, color = label)) + geom_point(size = 2) + labs(title = "OVERLAPPING", x ="X", y = "Y") + theme_bw() + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5, angle = 45),axis.text.y = element_text(vjust = 0.5, hjust = 0.5, angle = 45), panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank()))
      print(ggplot(data = overlapping_clusters, mapping = aes(x = x, y = y,fill = as.factor(label))) 
            + geom_point(colour = "white",size = 3,shape = 21) 
            + labs(title = "OVERLAPPING", x ="X", y = "Y") 
            + theme_bw() 
            + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5),axis.text.y = element_text(vjust = 0.5, hjust = 0.5))
            + theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank())
            + scale_fill_manual(values = c("#16048A","#9E189F"))
            # + coord_fixed(ratio = 0.6)
            + theme(axis.title.y = element_text(angle = 0, vjust = 0.5)))
      ggsave(paste(datasetpath, "overlapping_clusters.pdf", sep=""),device = cairo_pdf,width =5, height =3.75)
    }
    
    # Generate data set with normal distribution and noise 噪声点要多边界要不分明-我算法才凸显作用
    if (TRUE) {
      set.seed(1)
      # generateRingShapePointsbyRnom = function(xmean,ymean,num,r,sd,label){
      #   cluster = vector()
      #   for (i in 1:num) {
      #     angle = i * 18
      #     rest_angle = angle%%180
      #     if ( (rest_angle>=0 && rest_angle<90)) {
      #       r = r + 10
      #     } else {
      #       r = r - 5
      #     }
      #     # x = rnorm(1, mean = r * cos(angle) + xmean, sd = sd)
      #     # y = rnorm(1, mean = r * sin(angle) + ymean, sd = sd)
      #     x = runif(1, r * cos(angle) + xmean - sd, r * cos(angle) + xmean + sd)
      #     y = runif(1, r * sin(angle) + ymean - sd, r * sin(angle) + ymean + sd)
      #     cluster = rbind(cluster, data.frame(x,y,label))
      #   }
      #   return(cluster)
      # }
      # noise <- generateRingShapePointsbyRnom(500,500,20,50,1,'noise')
      # # x1 <- rnorm(20, mean = 465, sd = 10)
      # # y1 <- rnorm(20, mean = 510, sd = 10)
      # # x2 <- rnorm(20, mean = 515, sd = 10)
      # # y2 <- rnorm(20, mean = 520, sd = 10)
      # x1 = runif(40, 440, 490)
      # y1 = runif(40, 480, 530)
      # x2 = runif(40, 500, 550)
      # y2 = runif(40, 480, 530)
      
      # cluster1 <- data.frame(x = x1, y = y1, label = 'cluster 1')
      # cluster2 <- data.frame(x = x2, y = y2, label = 'cluster 2')
      # noise20_100_clusters <- rbind(cluster1, cluster2, noise)
      
      generateRingShapePointsbyRnom = function(xmean,ymean,num,r,sd,label){
        cluster = vector()
        for (i in 1:num) {
          angle = i * 18
          rest_angle = angle%%180
          if ( (rest_angle>=0 && rest_angle<90)) {
            r = r + 10
          } else {
            r = r - 5
          }
          x = round(runif(1, r * cos(angle) + xmean - sd, r * cos(angle) + xmean + sd))
          y = round(runif(1, r * sin(angle) + ymean - sd, r * sin(angle) + ymean + sd))
          cluster = rbind(cluster, data.frame(x,y,label))
        }
        return(cluster)
      }
      noise <- generateRingShapePointsbyRnom(500,500,20,100,20,'noise')
      x1 = round(runif(50, 460, 550))
      y1 = round(runif(50, 460, 550))
      cluster <- data.frame(x = x1, y = y1, label = 'cluster')
      noise20_100_clusters <- rbind(cluster, noise)
      
      equalidx = c()
      distdata = as.matrix(dist(noise20_100_clusters[,c(1:2)]))
      equalist = which(distdata==0, arr.ind = TRUE)
      for(i in c(1:nrow(equalist))){
        if(equalist[i,1] != equalist[i,2]){
          equalidx = c(equalidx, equalist[i,1], equalist[i,2])
        }
      }
      equalidx
      save(noise20_100_clusters, file=paste(datasetpath, "noise20_100_clusters.RData", sep=""))
      
      print(ggplot(data = noise20_100_clusters, mapping = aes(x = x, y = y,fill = as.factor(label))) 
            + geom_point(colour = "white",size = 3,shape = 21) 
            + labs(title = "NOISE", x ="X", y = "Y") 
            + theme_bw() 
            + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5),axis.text.y = element_text(vjust = 0.5, hjust = 0.5))
            + theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank())
            + scale_fill_manual(values = c("#16048A","#9E189F","#FDB330"))
            + theme(axis.title.y = element_text(angle = 0, vjust = 0.5)))
      ggsave(paste(datasetpath, "noise20_100_clusters.pdf", sep=""),device = cairo_pdf,width =5, height =3.75)
    }
    
    # Generate Ring data set
    if (TRUE) {
      set.seed(1)
      # generatePointsByRnom <- function(xmean, ymean, sd, num, label) {
      #   x <- rnorm(num, mean = xmean, sd = sd)
      #   y <- rnorm(num, mean = ymean, sd = sd)
      #   data.frame(x, y, label)
      # }
      # generateRingShapePointsbyRnom <- function(r, class) {
      #   cluster = vector()
      #   for (i in 1:50) {
      #     angle = i * 18
      #     x = r * cos(angle)
      #     y = r * sin(angle)
      #     cluster <- rbind(cluster, generatePointsByRnom(x+50, y+50, 1, 1, class))
      #   }
      #   cluster
      # }
      # cluster1 <- generateRingShapePointsbyRnom(10, 'cluster 1')
      # cluster2 <- generateRingShapePointsbyRnom(20, 'cluster 2')
      # ring_clusters <- rbind(cluster1, cluster2)
      
      generateRingShapePointsbyRnom = function(xmean,ymean,num,r,sd,label){
        cluster = vector()
        for (i in 1:num) {
          angle = i * 7
          # x = rnorm(1, mean = r * cos(angle) + xmean, sd = sd)
          # y = rnorm(1, mean = r * sin(angle) + ymean, sd = sd)
          x = runif(1, r * cos(angle) + xmean - sd, r * cos(angle) + xmean + sd)
          y = runif(1, r * sin(angle) + ymean - sd, r * sin(angle) + ymean + sd)
          cluster = rbind(cluster, data.frame(x,y,label))
        }
        return(cluster) 
      }
      cluster1 <- generateRingShapePointsbyRnom(50,50,100,5,1,'cluster 1')
      cluster2 <- generateRingShapePointsbyRnom(50,50,100,15,1,'cluster 2')
      ring_clusters <- rbind(cluster1, cluster2)
      # cluster1$x = cluster1$x
      # cluster1$y = cluster1$y
      # cluster2$x = cluster2$x
      # cluster2$y = cluster2$y
      equalidx = c()
      distdata = as.matrix(dist(noise20_100_clusters[,c(1:2)]))
      equalist = which(distdata==0, arr.ind = TRUE)
      for(i in c(1:nrow(equalist))){
        if(equalist[i,1] != equalist[i,2]){
          equalidx = c(equalidx, equalist[i,1], equalist[i,2])
        }
      }
      equalidx
      save(ring_clusters, file=paste(datasetpath, "ring_clusters.RData", sep=""))
      
      print(ggplot(data = ring_clusters, mapping = aes(x = x, y = y,fill = as.factor(label))) 
            + geom_point(colour = "white",size = 3,shape = 21) 
            + labs(title = "RING", x ="X", y = "Y") 
            + theme_bw() 
            + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5),axis.text.y = element_text(vjust = 0.5, hjust = 0.5))
            + theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank())
            + scale_fill_manual(values = c("#16048A","#9E189F"))
            + theme(axis.title.y = element_text(angle = 0, vjust = 0.5)))
      ggsave(paste(datasetpath, "ring_clusters.pdf", sep=""),device = cairo_pdf,width =5, height =3.75)
    }
    
    # Generate Half Ring data set
    # 对HDBSCAN，如果两个簇密度一样反而不好区别，还必须是密度不一样才行
    if (T) {
      set.seed(1)
      generatePointsByUniform <- function(x, y, num, label) {
        x <- round(runif(num,x-15,x+15))
        y <- round(runif(num, y-15,y+15))
        data.frame(x, y, label)
      }
      generateRingShapePointsbyUniform <- function(r,xoffset,yoffset,divangle, class) {
        cluster = vector()
        for (i in 1:60) {
          angle = i /divangle
          x = round(r * cos(angle)+xoffset)
          y = round(r * sin(angle)+yoffset)
          cluster <- rbind(cluster, generatePointsByUniform(x+100,y+100,1,class))
        }
        cluster
      }
      cluster1 <- generateRingShapePointsbyUniform(100,0,0,20,'cluster 1')
      cluster2 <- generateRingShapePointsbyUniform(100,100,50,-20,'cluster 2')
      halfring_clusters <- rbind(cluster1, cluster2)
      equalidx = c()
      distdata = as.matrix(dist(halfring_clusters[,c(1:2)]))
      equalist = which(distdata==0, arr.ind = TRUE)
      for(i in c(1:nrow(equalist))){
        if(equalist[i,1] != equalist[i,2]){
          equalidx = c(equalidx, equalist[i,1], equalist[i,2])
        }
      }
      equalidx
      save(halfring_clusters, file=paste(datasetpath, "halfring_clusters.RData", sep=""))
      
      print(ggplot(data = halfring_clusters, mapping = aes(x = x, y = y,fill = as.factor(label))) 
            + geom_point(colour = "white",size = 3,shape = 21) 
            + labs(title = "HALFRING", x ="X", y = "Y") 
            + theme_bw() 
            + theme(legend.position = "bottom",axis.text.x = element_text(vjust = 0.5, hjust = 0.5),axis.text.y = element_text(vjust = 0.5, hjust = 0.5))
            + theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), legend.direction = "horizontal",legend.title = element_blank())
            + scale_fill_manual(values = c("#16048A","#9E189F"))
            + theme(axis.title.y = element_text(angle = 0, vjust = 0.5)))
      ggsave(paste(datasetpath, "halfring_clusters.pdf", sep=""),device = cairo_pdf,width =5, height =3.75)
    }
  }
  #保存人工数据集到列表
  if (T) {
    data = overlapping_clusters
    data[which(data=="cluster 1", arr.ind = TRUE)] = 1
    data[which(data=="cluster 2", arr.ind = TRUE)] = 2
    data[,4] = as.numeric(data[,4])
    datalabel = data[,4]
    data = data[,1:3]
    idnum = c(1,2)
    idbin = c(3)
    idcat = NULL
    OVERLAPPING = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("OVERLAPPING")
    )
    
    data = noise20_100_clusters
    data[which(data=="cluster", arr.ind = TRUE)] = 1
    data[which(data=="noise", arr.ind = TRUE)] = 2
    data[,3] = as.numeric(data[,3])
    datalabel = data[,3]
    data = data[,1:2]
    idnum = c(1,2)
    idbin = NULL
    idcat = NULL
    NOISE = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("NOISE")
    )
    
    data = ring_clusters
    data[which(data=="cluster 1", arr.ind = TRUE)] = 1
    data[which(data=="cluster 2", arr.ind = TRUE)] = 2
    data[,3] = as.numeric(data[,3])
    datalabel = data[,3]
    data = data[,1:2]
    idnum = c(1,2)
    idbin = NULL
    idcat = NULL
    RING = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("RING")
    )
    
    data = halfring_clusters
    data[which(data=="cluster 1", arr.ind = TRUE)] = 1
    data[which(data=="cluster 2", arr.ind = TRUE)] = 2
    data[,3] = as.numeric(data[,3])
    datalabel = data[,3]
    data = data[,1:2]
    idnum = c(1,2)
    idbin = NULL
    idcat = NULL
    HALFRING = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("HALFRING")
    )
  }
  #保存真实数据集到列表
  if (T) {
    data = datasets::iris
    data$Species = as.numeric(data$Species)
    datalabel = data$Species
    data = data[,1:4]
    data = scale(data, center = TRUE, scale = TRUE)
    idnum = c(1:4)
    idbin = NULL
    idcat = NULL
    IRIS = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("IRIS")
    )
    #http://archive.ics.uci.edu/dataset/53/iris
    
    #59 71 48, so exclude
    data = wine
    datalabel = as.numeric(data$Wine)
    data = data[,1:13]
    idnum = c(1:13)
    idbin = NULL
    idcat = NULL
    WINE = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("WINE")
    )
    #http://archive.ics.uci.edu/dataset/109/wine
    
    # 88 12,so exclude
    fertility_Diagnosis = read.csv(paste(datasetpath, "fertility_Diagnosis.txt", sep=""), header=F)
    data = fertility_Diagnosis
    data[which(data=="N", arr.ind = TRUE)] = 1
    data[which(data=="O", arr.ind = TRUE)] = 2
    datalabel = as.numeric(data[,10])
    data = data[,1:9]
    idnum = c(2,9)
    idbin = c(3,4,5)
    idcat = c(1,6,7,8)
    FERTILITY = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("FERTILITY")
    )
    #http://archive.ics.uci.edu/dataset/244/fertility
    
    tae = read.csv(paste(datasetpath, "tae.data", sep=""), header=F)
    data = tae
    datalabel = as.numeric(data[,6])
    data = data[,1:5]
    idnum = c(5)
    idbin = c(1,4)
    idcat = c(2,3)
    TAE = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("TAE")
    )
    #http://archive.ics.uci.edu/dataset/100/teaching+assistant+evaluation
    
    #40 60 36  8  4 27 15  4,so e
    flag = read.csv(paste(datasetpath, "flag.data", sep=""), header=F)
    data = flag[,c(2:ncol(flag))]
    type18 = unique(flag[,18])
    type29 = unique(flag[,29])
    type30 = unique(flag[,30])
    #可以发现颜色通用，所以用一个循环就行
    for(i in c(1:length(type18))){
      data[which(data==type18[i], arr.ind = TRUE)] = i
    }
    # for(i in c(1:length(type29))){
    #   data[which(data==type29[i], arr.ind = TRUE)] = i
    # }
    # for(i in c(1:length(type30))){
    #   data[which(data==type30[i], arr.ind = TRUE)] = i
    # }
    datalabel = data[,6] + 1
    for(i in c(1:29)){data[,i] = as.numeric(data[,i])}
    idnum = c(3,4,7,8,9,18,19,20,21,22)
    idbin = c(10:16,23:27)
    idcat = c(1,2,5,17,28,29)#6是宗教当作预测标签了
    FLAG = list(
      data = data,
      datalabel = datalabel,
      idnum = idnum,
      idbin = idbin,
      idcat = idcat,
      dataname = c("FLAG")
    )
  }
  #保存神经影像数据集到列表 cobre不能用，因为我降维是将病人正常人一起做的；之前将年龄也作为一个特征属于胡闹了
  if (T) {
    results_path = paste(datasetpath, "project_sald/", sep="")
    pca_data = read.table(paste(results_path, "sald_pre_data.txt", sep=""), header=F)
    pca_anat = read.table(paste(results_path, "sald_pre_anat.txt", sep=""), header=F)
    pca_func = read.table(paste(results_path, "sald_pre_func.txt", sep=""), header=F)
    pca_sex = read.table(paste(results_path, "sald_pre_sex.txt", sep=""), header=F)#好像加上是反效果了？
    pca_label = read.table(paste(results_path, "sald_pre_label.txt", sep=""), header=F)
    data = cbind(pca_anat,pca_func,pca_sex)
    SALD = list(
      data = data,
      datalabel = c(t(pca_label)),
      idnum = c(1: (ncol(data)-1) ),
      idbin = c(ncol(data)),
      idcat = NULL,
      dataname = c("SALD")
    )
  }
  
  datalist = list(
    
    
    
    NOISE=NOISE,#1-2-3 50
    OVERLAPPING=OVERLAPPING,
     #1-2 50
    RING=RING,#1-2 100
    HALFRING=HALFRING,#1-2 120
    FERTILITY=FERTILITY, #1-2 88-12
    
    FLAG=FLAG, #1-2-3-4-5-6-7-8 40-60-36-8-4-27-15-4
    # TAE=TAE, #1-2-3 49-50-52 50 主要是数值变量只有一个，我算法中还要加入把向量变矩阵的，太麻烦;改了也不行，因为MCPLDA算法处理不了
    
    # WINE=WINE, #1-2-3 59-71-48
    # IRIS=IRIS,#1-2-3 50
    
    
    # ZOO=ZOO, #1-2-3-4-5-6-7 41-20-5-13-4-8-10
    
    
    SALD = SALD
  )
  return(datalist)
}
distknn = function(distdata, datalabel, train_idx, test_idx, k){
  distdata[-train_idx, -train_idx] = Inf
  diag(distdata) = Inf
  predictions = rep(0, length(test_idx))
  
  for (i in c(1:length(test_idx))) {
    ordered_neighbors = order(distdata[test_idx[i], ])
    indices = ordered_neighbors[1:k]
    label_q = datalabel[indices]
    table_q = table(label_q)
    vote = rep(0, length(table_q))
    for (p in c(1:length(table_q))) {
      distdata_colnames = intersect(indices, which(datalabel==names(table_q)[p]))
      vote[p] = sum(1/exp(distdata[test_idx[i], distdata_colnames]))
    }
    predictions[i] = names(table_q)[which.max(vote)]
  }
  datalabel[test_idx] = as.numeric(predictions)
  return(datalabel)
}
