你是安卓恶意软件分析专家。下面是一个 APK 的模型预测结果与可解释性证据，包括 LightGBM 的局部/全局 SHAP 特征，以及图模型的 attention 节点和边。
请你基于这些证据自己理解权限、API、方法名、类名或其他特征的用途，不要依赖预设标签。你的目标是解释：这些证据可能代表什么安卓行为，为什么是恶意的软件
要求：
1. 不要把 SHAP 或 attention 写成绝对因果证明；使用“模型关注到”“可能说明”“需要结合上下文确认”等表述。
2. 优先解释正向 SHAP 特征、图 attention Top 节点/边；同时说明负向 SHAP 是否削弱恶意判断。
3. 对你看得懂的 API/权限/方法名，简要解释其通常用途；对混淆名、未知名或证据不足处，明确说不确定。
4. 如果判定为恶意软件则输出：最终判断、关键证据解释、可能行为、防护建议（给非专业人员)。
5. 如果判定为良性软件则输出：最终判断、关键证据解释。
5. 解释内容尽量简洁，但又能清楚表达。

证据 JSON：
```json
{
  "prediction": {
    "apk_name": "org.ooma.oomaapp.apk",
    "final_prediction": "benign",
    "final_prob_malware": 0.0004512476483876,
    "modalities": {
      "ensemble": {
        "prob_malware": 0.0004512476483876,
        "pred": "benign",
        "prob_static": 0.000122230740833,
        "prob_graph": 0.0007802645559422
      },
      "static": {
        "prob_malware": 0.00012223074083301643,
        "pred": "benign"
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
        "rank": 9,
        "name": "opcode:filled-new-array",
        "value": "6.315358001522335",
        "shap_value": 0.309337982046123,
        "abs_shap_value": 0.309337982046123,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 10,
        "name": "api:android.content.res.ColorStateList.getDefaultColor",
        "value": "4.060443010546419",
        "shap_value": 0.30236172984530735,
        "abs_shap_value": 0.30236172984530735,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 11,
        "name": "api:android.view.Window.getDecorView",
        "value": "4.219507705176107",
        "shap_value": 0.258698961274014,
        "abs_shap_value": 0.258698961274014,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 12,
        "name": "api:android.view.View.hasOnClickListeners",
        "value": "1.0986122886681096",
        "shap_value": 0.2538883611442327,
        "abs_shap_value": 0.2538883611442327,
        "direction": "pushes_toward_malware"
      },
      {
        "rank": 15,
        "name": "api:android.os.BadParcelableException.<init>",
        "value": "1.3862943611198906",
        "shap_value": 0.2375519592675156,
        "abs_shap_value": 0.2375519592675156,
        "direction": "pushes_toward_malware"
      }
    ],
    "top_negative": [
      {
        "rank": 1,
        "name": "api:mono.android.net.wifi.p2p.WifiP2pManager_ConnectionInfoListenerImplementor.n_onConnectionInfoAvailable",
        "value": "0.6931471805599453",
        "shap_value": -3.651809513663169,
        "abs_shap_value": 3.651809513663169,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 2,
        "name": "api_pkg:mono.android.media",
        "value": "4.060443010546419",
        "shap_value": -0.9302700595787108,
        "abs_shap_value": 0.9302700595787108,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 3,
        "name": "suspicious_string:ip",
        "value": "5.746203190540153",
        "shap_value": -0.9125031500542412,
        "abs_shap_value": 0.9125031500542412,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 4,
        "name": "api:com.stub.StubApp.interface5",
        "value": "0.0",
        "shap_value": -0.44879468879509743,
        "abs_shap_value": 0.44879468879509743,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 5,
        "name": "api:mono.android.net.sip.SipRegistrationListenerImplementor.n_onRegistrationDone",
        "value": "0.6931471805599453",
        "shap_value": -0.4354358001699154,
        "abs_shap_value": 0.4354358001699154,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 6,
        "name": "api:android.widget.LinearLayout.<init>",
        "value": "4.23410650459726",
        "shap_value": -0.3624002594700184,
        "abs_shap_value": 0.3624002594700184,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 7,
        "name": "api:android.widget.EditText.<init>",
        "value": "2.302585092994046",
        "shap_value": -0.3388711908499356,
        "abs_shap_value": 0.3388711908499356,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 8,
        "name": "api:java.lang.String.lastIndexOf",
        "value": "2.5649493574615367",
        "shap_value": -0.32211054139775175,
        "abs_shap_value": 0.32211054139775175,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 13,
        "name": "api:com.qihoo.util.a.<init>",
        "value": "0.0",
        "shap_value": -0.24090519126160584,
        "abs_shap_value": 0.24090519126160584,
        "direction": "pushes_toward_benign"
      },
      {
        "rank": 14,
        "name": "api:android.graphics.drawable.Drawable.invalidateSelf",
        "value": "4.672828834461906",
        "shap_value": -0.23876230848075722,
        "abs_shap_value": 0.23876230848075722,
        "direction": "pushes_toward_benign"
      }
    ]
  },
  "static_shap_global_top": [],
  "graph_attention": {
    "top_nodes": [
      {
        "rank": 1,
        "type": "package",
        "name": "io.github.inflationx.viewpump",
        "attention": 0.750000000000113,
        "normalized_attention": 0.0013202988110953262
      },
      {
        "rank": 2,
        "type": "package",
        "name": "com.microsoft.appcenter.crashes.utils",
        "attention": 0.75,
        "normalized_attention": 0.0013202988110951274
      },
      {
        "rank": 3,
        "type": "package",
        "name": "androidx.emoji2.text",
        "attention": 0.75,
        "normalized_attention": 0.0013202988110951274
      },
      {
        "rank": 4,
        "type": "intent",
        "name": "android.intent.category.LAUNCHER",
        "attention": 0.7499651717957994,
        "normalized_attention": 0.0013202374995796625
      },
      {
        "rank": 5,
        "type": "intent",
        "name": "android.intent.action.BOOT_COMPLETED",
        "attention": 0.7433831354183127,
        "normalized_attention": 0.0013086504931746218
      },
      {
        "rank": 6,
        "type": "intent",
        "name": "androidx.profileinstaller.action.SAVE_PROFILE",
        "attention": 0.6874614288008161,
        "normalized_attention": 0.0012102060094926334
      },
      {
        "rank": 7,
        "type": "package",
        "name": "com.makeramen.roundedimageview",
        "attention": 0.6666666666667266,
        "normalized_attention": 0.0011735989431957742
      },
      {
        "rank": 8,
        "type": "package",
        "name": "android.view.accessibility",
        "attention": 0.6666600555409635,
        "normalized_attention": 0.001173587304980461
      },
      {
        "rank": 9,
        "type": "opcode",
        "name": "rem-float",
        "attention": 0.6510080024600029,
        "normalized_attention": 0.001146033455548474
      },
      {
        "rank": 10,
        "type": "opcode",
        "name": "iget-byte",
        "attention": 0.57173041254282,
        "normalized_attention": 0.001006473311929616
      },
      {
        "rank": 11,
        "type": "api",
        "name": "androidx.core.util.Preconditions.checkState",
        "attention": 0.5602269681113466,
        "normalized_attention": 0.0009862226665877851
      },
      {
        "rank": 12,
        "type": "package",
        "name": "android.transition",
        "attention": 0.5359279662370682,
        "normalized_attention": 0.0009434467422072409
      },
      {
        "rank": 13,
        "type": "intent",
        "name": "android.intent.action.VIEW",
        "attention": 0.5337431599331361,
        "normalized_attention": 0.0009396006126531681
      },
      {
        "rank": 14,
        "type": "opcode",
        "name": "iput-char",
        "attention": 0.5094392895698547,
        "normalized_attention": 0.0008968161177923007
      },
      {
        "rank": 15,
        "type": "permission",
        "name": "android.permission.VIBRATE",
        "attention": 0.5001829032015053,
        "normalized_attention": 0.0008805211899027421
      },
      {
        "rank": 16,
        "type": "opcode",
        "name": "sput-byte",
        "attention": 0.5000658470671624,
        "normalized_attention": 0.0008803151244694031
      },
      {
        "rank": 17,
        "type": "package",
        "name": "androidx.constraintlayout.motion.utils",
        "attention": 0.5000000013308856,
        "normalized_attention": 0.0008801992097396405
      },
      {
        "rank": 18,
        "type": "package",
        "name": "android.content.pm",
        "attention": 0.5,
        "normalized_attention": 0.0008801992073967516
      },
      {
        "rank": 19,
        "type": "package",
        "name": "java.util.logging",
        "attention": 0.5,
        "normalized_attention": 0.0008801992073967516
      },
      {
        "rank": 20,
        "type": "package",
        "name": "kotlinx.coroutines.debug.internal",
        "attention": 0.4999999098362859,
        "normalized_attention": 0.0008801990486726922
      },
      {
        "rank": 21,
        "type": "package",
        "name": "android.text.util",
        "attention": 0.4999879219358263,
        "normalized_attention": 0.0008801779451917264
      },
      {
        "rank": 22,
        "type": "intent",
        "name": "androidx.profileinstaller.action.SKIP_FILE",
        "attention": 0.4993655573671276,
        "normalized_attention": 0.0008790823355915656
      },
      {
        "rank": 23,
        "type": "component",
        "name": "provider:mono.MonoRuntimeProvider",
        "attention": 0.4993013579937724,
        "normalized_attention": 0.0008789693191164804
      },
      {
        "rank": 24,
        "type": "permission",
        "name": "android.permission.WAKE_LOCK",
        "attention": 0.4992226008325815,
        "normalized_attention": 0.0008788306751347663
      },
      {
        "rank": 25,
        "type": "intent",
        "name": "androidx.profileinstaller.action.INSTALL_PROFILE",
        "attention": 0.4981903846946807,
        "normalized_attention": 0.0008770135634818814
      }
    ],
    "top_edges": [
      {
        "rank": 1,
        "source_type": "app",
        "source": "org.ooma.oomaapp|org.ooma.oomaapp",
        "target_type": "class",
        "target": "androidx.recyclerview.widget.RecyclerView$ViewFlinger",
        "attention": 1.0
      },
      {
        "rank": 2,
        "source_type": "app",
        "source": "org.ooma.oomaapp|org.ooma.oomaapp",
        "target_type": "class",
        "target": "androidx.constraintlayout.core.motion.utils.TypedValues$CycleType$-CC",
        "attention": 1.0
      },
      {
        "rank": 3,
        "source_type": "app",
        "source": "org.ooma.oomaapp|org.ooma.oomaapp",
        "target_type": "class",
        "target": "androidx.appcompat.widget.ResourceManagerInternal",
        "attention": 1.0
      },
      {
        "rank": 4,
        "source_type": "method",
        "source": "com.caverock.androidsvg.SVGParser$ColourKeywords.<clinit>()V",
        "target_type": "class",
        "target": "com.caverock.androidsvg.SVGParser$ColourKeywords",
        "attention": 1.0
      },
      {
        "rank": 5,
        "source_type": "method",
        "source": "com.microsoft.appcenter.channel.DefaultChannel.enqueue(Lcom/microsoft/appcenter/ingestion/models/Log; Ljava/lang/String; I)V",
        "target_type": "class",
        "target": "com.microsoft.appcenter.channel.DefaultChannel",
        "attention": 1.0
      },
      {
        "rank": 6,
        "source_type": "method",
        "source": "androidx.constraintlayout.core.parser.CLParser.parse()Landroidx/constraintlayout/core/parser/CLObject;",
        "target_type": "class",
        "target": "androidx.constraintlayout.core.parser.CLParser",
        "attention": 1.0
      },
      {
        "rank": 7,
        "source_type": "method",
        "source": "kotlinx.serialization.SerializersKt__SerializersKt.builtinParametrizedSerializer$SerializersKt__SerializersKt(Lkotlin/reflect/KClass; Ljava/util/List; Lkotlin/jvm/functions/Function0;)Lkotlinx/serialization/KSerializer;",
        "target_type": "class",
        "target": "kotlinx.serialization.SerializersKt__SerializersKt",
        "attention": 1.0
      },
      {
        "rank": 8,
        "source_type": "method",
        "source": "androidx.customview.widget.ExploreByTouchHelper.createNodeForChild(I)Landroidx/core/view/accessibility/AccessibilityNodeInfoCompat;",
        "target_type": "class",
        "target": "androidx.customview.widget.ExploreByTouchHelper",
        "attention": 1.0
      },
      {
        "rank": 9,
        "source_type": "method",
        "source": "androidx.savedstate.serialization.SavedStateEncoder_androidKt.encodeFormatSpecificTypesOnPlatform(Landroidx/savedstate/serialization/SavedStateEncoder; Lkotlinx/serialization/SerializationStrategy; Ljava/lang/Object;)Z",
        "target_type": "class",
        "target": "androidx.savedstate.serialization.SavedStateEncoder_androidKt",
        "attention": 1.0
      },
      {
        "rank": 10,
        "source_type": "method",
        "source": "kotlinx.coroutines.internal.ExceptionsConstructorKt.createConstructor(Ljava/lang/Class;)Lkotlin/jvm/functions/Function1;",
        "target_type": "class",
        "target": "kotlinx.coroutines.internal.ExceptionsConstructorKt",
        "attention": 1.0
      },
      {
        "rank": 11,
        "source_type": "method",
        "source": "androidx.savedstate.serialization.SavedStateCodecUtils_androidKt.<clinit>()V",
        "target_type": "class",
        "target": "androidx.savedstate.serialization.SavedStateCodecUtils_androidKt",
        "attention": 1.0
      },
      {
        "rank": 12,
        "source_type": "method",
        "source": "kotlin.text.CharDirectionality.<clinit>()V",
        "target_type": "class",
        "target": "kotlin.text.CharDirectionality",
        "attention": 1.0
      },
      {
        "rank": 13,
        "source_type": "method",
        "source": "androidx.savedstate.serialization.SavedStateEncoder.encodeFormatSpecificTypes(Lkotlinx/serialization/SerializationStrategy; Ljava/lang/Object;)Z",
        "target_type": "class",
        "target": "androidx.savedstate.serialization.SavedStateEncoder",
        "attention": 1.0
      },
      {
        "rank": 14,
        "source_type": "method",
        "source": "androidx.collection.MutableObjectIntMap.findIndex(Ljava/lang/Object;)I",
        "target_type": "class",
        "target": "androidx.collection.MutableObjectIntMap",
        "attention": 1.0
      },
      {
        "rank": 15,
        "source_type": "method",
        "source": "androidx.savedstate.serialization.SavedStateDecoder.decodeFormatSpecificTypes(Lkotlinx/serialization/DeserializationStrategy;)Ljava/lang/Object;",
        "target_type": "class",
        "target": "androidx.savedstate.serialization.SavedStateDecoder",
        "attention": 1.0
      },
      {
        "rank": 16,
        "source_type": "method",
        "source": "androidx.constraintlayout.widget.ConstraintSet$Transform.fillFromAttributeList(Landroid/content/Context; Landroid/util/AttributeSet;)V",
        "target_type": "class",
        "target": "androidx.constraintlayout.widget.ConstraintSet$Transform",
        "attention": 1.0
      },
      {
        "rank": 17,
        "source_type": "method",
        "source": "com.flaviofaria.kenburnsview.RandomTransitionGenerator.generateRandomRect(Landroid/graphics/RectF; Landroid/graphics/RectF;)Landroid/graphics/RectF;",
        "target_type": "class",
        "target": "com.flaviofaria.kenburnsview.RandomTransitionGenerator",
        "attention": 1.0
      },
      {
        "rank": 18,
        "source_type": "method",
        "source": "com.google.android.material.slider.BaseSlider.onKeyDown(I Landroid/view/KeyEvent;)Z",
        "target_type": "class",
        "target": "com.google.android.material.slider.BaseSlider",
        "attention": 1.0
      },
      {
        "rank": 19,
        "source_type": "method",
        "source": "androidx.core.widget.NestedScrollView$AccessibilityDelegate.performAccessibilityAction(Landroid/view/View; I Landroid/os/Bundle;)Z",
        "target_type": "class",
        "target": "androidx.core.widget.NestedScrollView$AccessibilityDelegate",
        "attention": 1.0
      },
      {
        "rank": 20,
        "source_type": "method",
        "source": "com.caverock.androidsvg.CSSParser$MediaType.<clinit>()V",
        "target_type": "class",
        "target": "com.caverock.androidsvg.CSSParser$MediaType",
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
