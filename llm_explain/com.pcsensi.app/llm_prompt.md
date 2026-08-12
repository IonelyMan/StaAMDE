你是安卓恶意软件分析专家。下面是一个 APK 的模型预测结果与可解释性证据，包括 LightGBM 的局部/全局 SHAP 特征，以及图模型的 attention 节点和边。
请你基于这些证据自己理解权限、API、方法名、类名或其他特征的用途，不要依赖预设标签。你的目标是解释：这些证据可能代表什么安卓行为，为什么是恶意的软件
要求：
1. 不要把 SHAP 或 attention 写成绝对因果证明；使用“模型关注到”“可能说明”“需要结合上下文确认”等表述。
2. 优先解释正向 SHAP 特征、图 attention Top 节点/边；同时说明负向 SHAP 是否削弱恶意判断。
3. 对你看得懂的 API/权限/方法名，简要解释其通常用途；对混淆名、未知名或证据不足处，明确说不确定。
4. 如果判定为恶意软件则输出：最终判断、关键证据解释、可能行为、防护建议（给非专业人员)。
5. 如果判定为良性软件则输出：最终判断、关键证据解释。
6. 解释内容尽量简洁，但又能清楚表达。

证据 JSON：
```json
{
  "prediction": {
    "apk_name": "com.pcsensi.app.apk",
    "final_prediction": "malware",
    "final_prob_malware": 0.9983318725650374,
    "modalities": {
      "ensemble": {
        "prob_malware": 0.9983318725650374,
        "pred": "malware",
        "prob_static": 0.9999671537528164,
        "prob_graph": 0.9966965913772584
      },
      "static": {
        "prob_malware": 0.9999671537528163,
        "pred": "malware"
      },
      "graph": {
        "prob_malware": 0.9966965913772583,
        "pred": "malware"
      }
    }
  },
  "evidence_semantics": {
    "positive_shap": "positive SHAP values increase the malware-class score",
    "negative_shap": "negative SHAP values decrease the malware-class score",
    "graph_attention": "higher attention means the graph model focused more on that node/edge"
  },
  "static_shap_local": {
    "top_positive": [
      {
        "rank": 1,
        "name": "api:android.view.Window.getDecorView",
        "value": "4.304065093204169",
        "shap_value": 0.726768418892123,
        "abs_shap_value": 0.726768418892123,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 2,
        "name": "api:mono.android.net.wifi.p2p.WifiP2pManager_ConnectionInfoListenerImplementor.n_onConnectionInfoAvailable",
        "value": "0.0",
        "shap_value": 0.7264536692283147,
        "abs_shap_value": 0.7264536692283147,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 3,
        "name": "api:android.graphics.drawable.GradientDrawable.<init>",
        "value": "5.003946305945459",
        "shap_value": 0.6922706524177489,
        "abs_shap_value": 0.6922706524177489,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 4,
        "name": "opcode:nop",
        "value": "7.60489448081162",
        "shap_value": 0.6432329519197426,
        "abs_shap_value": 0.6432329519197426,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 5,
        "name": "opcode:sparse-switch",
        "value": "5.638354669333745",
        "shap_value": 0.6302272412964726,
        "abs_shap_value": 0.6302272412964726,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 6,
        "name": "api:java.util.ArrayList.removeAll",
        "value": "1.791759469228055",
        "shap_value": 0.5103659914946655,
        "abs_shap_value": 0.5103659914946655,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 7,
        "name": "api:android.widget.EditText.<init>",
        "value": "1.0986122886681096",
        "shap_value": 0.4229223090873092,
        "abs_shap_value": 0.4229223090873092,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 8,
        "name": "api:java.lang.Throwable.<init>",
        "value": "3.332204510175204",
        "shap_value": 0.39011724185267294,
        "abs_shap_value": 0.39011724185267294,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 9,
        "name": "api:android.os.Bundle.putCharSequence",
        "value": "3.5263605246161616",
        "shap_value": 0.3887066709818487,
        "abs_shap_value": 0.3887066709818487,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 10,
        "name": "api:java.nio.channels.FileChannel.map",
        "value": "1.6094379124341003",
        "shap_value": 0.33564956657131245,
        "abs_shap_value": 0.33564956657131245,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 11,
        "name": "api:java.io.ByteArrayOutputStream.write",
        "value": "3.970291913552122",
        "shap_value": 0.3164053605275666,
        "abs_shap_value": 0.3164053605275666,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 12,
        "name": "api:android.content.pm.PackageManager.getActivityInfo",
        "value": "1.791759469228055",
        "shap_value": 0.2966936211463909,
        "abs_shap_value": 0.2966936211463909,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 13,
        "name": "api:java.lang.Integer.toOctalString",
        "value": "0.0",
        "shap_value": 0.28482663003680336,
        "abs_shap_value": 0.28482663003680336,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 15,
        "name": "suspicious_string:ip",
        "value": "1.0986122886681096",
        "shap_value": 0.23332605625595668,
        "abs_shap_value": 0.23332605625595668,
        "direction": "pushes_toward_malware"
      }
    ],
    "top_negative": [
      {
        "rank": 14,
        "name": "api:com.stub.StubApp.interface5",
        "value": "0.0",
        "shap_value": -0.2388803849632692,
        "abs_shap_value": 0.2388803849632692,
        "direction": "pushes_toward_benign"
      }
    ]
  },
  "graph_attention": {
    "top_nodes": [
      {
        "rank": 1,
        "type": "intent",
        "name": "android.intent.action.MAIN",
        "attention": 0.6529546605888754,
        "normalized_attention": 0.001176605008755002
      },
      {
        "rank": 2,
        "type": "opcode",
        "name": "ushr-long/2addr",
        "attention": 0.5748146655387245,
        "normalized_attention": 0.001035799046091099
      },
      {
        "rank": 3,
        "type": "api",
        "name": "android.graphics.Color.red",
        "attention": 0.5413945357397708,
        "normalized_attention": 0.0009755769594928843
      },
      {
        "rank": 4,
        "type": "component",
        "name": "activity:.DebugActivity",
        "attention": 0.5342393306054873,
        "normalized_attention": 0.0009626834912204092
      },
      {
        "rank": 5,
        "type": "api",
        "name": "com.bumptech.glide.gifdecoder.GifHeaderParser.read",
        "attention": 0.5164245752200713,
        "normalized_attention": 0.0009305818283004727
      },
      {
        "rank": 6,
        "type": "package",
        "name": "javax.net.ssl",
        "attention": 0.5000000000075666,
        "normalized_attention": 0.0009009852289833354
      },
      {
        "rank": 7,
        "type": "package",
        "name": "com.bumptech.glide.load.model",
        "attention": 0.5000000000000007,
        "normalized_attention": 0.0009009852289697018
      },
      {
        "rank": 8,
        "type": "package",
        "name": "com.bumptech.glide.load.data",
        "attention": 0.5000000000000001,
        "normalized_attention": 0.0009009852289697008
      },
      {
        "rank": 9,
        "type": "package",
        "name": "com.google.android.gms.common.api",
        "attention": 0.5,
        "normalized_attention": 0.0009009852289697006
      },
      {
        "rank": 10,
        "type": "component",
        "name": "activity:.DisplaytesterActivity",
        "attention": 0.4999904366318333,
        "normalized_attention": 0.0009009679960627858
      },
      {
        "rank": 11,
        "type": "opcode",
        "name": "sput-byte",
        "attention": 0.49852091004140675,
        "normalized_attention": 0.0008983199525596807
      },
      {
        "rank": 12,
        "type": "permission",
        "name": "android.permission.ACCESS_NETWORK_STATE",
        "attention": 0.4874431248754263,
        "normalized_attention": 0.0008783581109511846
      },
      {
        "rank": 13,
        "type": "intent",
        "name": "com.pcsensi.app.START_BACKGROUND",
        "attention": 0.4607546180486679,
        "normalized_attention": 0.000830266210082852
      },
      {
        "rank": 14,
        "type": "opcode",
        "name": "rem-float",
        "attention": 0.4558132859528996,
        "normalized_attention": 0.0008213620756234097
      },
      {
        "rank": 15,
        "type": "api",
        "name": "android.view.View.setElevation",
        "attention": 0.449697161607054,
        "normalized_attention": 0.000810341000235112
      },
      {
        "rank": 16,
        "type": "api",
        "name": "android.view.View.setTranslationZ",
        "attention": 0.44681812789764574,
        "normalized_attention": 0.0008051530665433466
      },
      {
        "rank": 17,
        "type": "component",
        "name": "activity:.SpeakercleanerActivity",
        "attention": 0.4386149756691594,
        "normalized_attention": 0.0007903712285656345
      },
      {
        "rank": 18,
        "type": "component",
        "name": "activity:.MoreActivity",
        "attention": 0.43269917369368294,
        "normalized_attention": 0.0007797111281708063
      },
      {
        "rank": 19,
        "type": "class",
        "name": "com.google.android.gms.internal.ads.zzcrq",
        "attention": 0.4323984781901042,
        "normalized_attention": 0.0007791692837565222
      },
      {
        "rank": 20,
        "type": "permission",
        "name": "android.permission.READ_EXTERNAL_STORAGE",
        "attention": 0.4318769723273088,
        "normalized_attention": 0.0007782295455981227
      },
      {
        "rank": 21,
        "type": "component",
        "name": "activity:.OngameActivity",
        "attention": 0.4302575961514735,
        "normalized_attention": 0.0007753114775689767
      },
      {
        "rank": 22,
        "type": "package",
        "name": "com.google.android.material.datepicker",
        "attention": 0.42857131147556454,
        "normalized_attention": 0.0007722728423993128
      },
      {
        "rank": 23,
        "type": "opcode",
        "name": "iput-char",
        "attention": 0.4124398257602575,
        "normalized_attention": 0.0007432043816976581
      },
      {
        "rank": 24,
        "type": "class",
        "name": "com.google.android.gms.internal.ads.zzdtk",
        "attention": 0.3994915201988891,
        "normalized_attention": 0.0007198719175956998
      },
      {
        "rank": 25,
        "type": "package",
        "name": "androidx.core.content.pm",
        "attention": 0.3990898482326884,
        "normalized_attention": 0.0007191481165788236
      }
    ],
    "top_edges": [
      {
        "rank": 1,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "permission",
        "target": "android.permission.WRITE_SETTINGS",
        "attention": 1.0
      },
      {
        "rank": 2,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "component",
        "target": "activity:.OptimizeActivity",
        "attention": 1.0
      },
      {
        "rank": 3,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzdqo",
        "attention": 1.0
      },
      {
        "rank": 4,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzavp",
        "attention": 1.0
      },
      {
        "rank": 5,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzee",
        "attention": 1.0
      },
      {
        "rank": 6,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzef",
        "attention": 1.0
      },
      {
        "rank": 7,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzeo",
        "attention": 1.0
      },
      {
        "rank": 8,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzem",
        "attention": 1.0
      },
      {
        "rank": 9,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzel",
        "attention": 1.0
      },
      {
        "rank": 10,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzej",
        "attention": 1.0
      },
      {
        "rank": 11,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzen",
        "attention": 1.0
      },
      {
        "rank": 12,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzep",
        "attention": 1.0
      },
      {
        "rank": 13,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzeh",
        "attention": 1.0
      },
      {
        "rank": 14,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzei",
        "attention": 1.0
      },
      {
        "rank": 15,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzek",
        "attention": 1.0
      },
      {
        "rank": 16,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzeg",
        "attention": 1.0
      },
      {
        "rank": 17,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzbjf",
        "attention": 1.0
      },
      {
        "rank": 18,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzaeq",
        "attention": 1.0
      },
      {
        "rank": 19,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.measurement.internal.zzy",
        "attention": 1.0
      },
      {
        "rank": 20,
        "source_type": "app",
        "source": "com.pcsensi.app|com.pcsensi.app",
        "target_type": "class",
        "target": "com.google.android.gms.internal.ads.zzbjg",
        "attention": 1.0
      }
    ]
  },
  "caveats": [
    "SHAP/attention are model evidence, not causal proof.",
    "Obfuscation or missing extraction can reduce confidence."
  ]
}
```
