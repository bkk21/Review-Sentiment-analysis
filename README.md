# 제품 review 데이터의 감성 분석 

## 1. 데이터 전처리
- `amazon_uk_shoes_products_dataset_2021_12.csv` 데이터 사용
<img width="1811" alt="image" src="https://github.com/user-attachments/assets/145b7d4f-2aed-48ae-b15a-d3d2b92520cc" />


### data info 확인
<img width="367" alt="image" src="https://github.com/user-attachments/assets/b7a05357-2a3d-44d0-bf71-a569db3f7a6f" />

### sample data 확인 및 분석
```python
#샘플 데이터 출력
print("url : ", row_data.iloc[0, 0])
print("product_name : ", row_data.iloc[0, 1])
print("reviewer_name : ", row_data.iloc[0, 2])
print("review_title : ", row_data.iloc[0, 3])
print("review_text : ", row_data.iloc[0, 4])
print("review_rating : ", row_data.iloc[0, 5])
print("verified_purchase : ", row_data.iloc[0, 6])
print("review_date : ", row_data.iloc[0, 7])
print("helpful_count : ", row_data.iloc[0, 8])
print("uniq_id : ", row_data.iloc[0, 9])
print("scraped_at : ", row_data.iloc[0, 10])
```
1. url - 상품 링크<br>
3. product_name - 상품 이름<br>
4. reviewer_name - 리뷰 작성자 이름<br>
5. review_title - 리뷰 제목<br>
6. review_text - 리뷰 내용<br>
7. review_rating - 리뷰 별점 값<br>
8. verified_purchase - 구매 여부<br>
9. review_date - 리뷰 작성일<br>
10. helpful_count - 도움 되었다고 누른 사람의 수<br>
11. uniq_id - 리뷰 아이디<br>
12. scraped_at - 리뷰 스크랩 날짜

### 사용할 데이터 선택
```python
df = row_data[['review_title', 'review_text', 'review_rating', 'uniq_id']]
```
1. review_title - 리뷰 제목으로, 제목은 내용을 요약한다고 생각하여 포함하였습니다.<br>
2. review_text - 리뷰 내용으로, 감정이 가장 잘 드러나는 열이라고 생각합니다.<br>
3. reveiw_rating - 감정 분석 후 맞게 분석을 하였는지 확인하는 지표로 사용됩니다.<br>
4. uniq_id - 데이터의 중복을 제거하기 위해 사용됩니다.

### 중복 데이터 확인 및 제거
```python
#전체 데이터 개수 확인
print("리뷰 개수 :", len(df['uniq_id']))

#중복 개수 확인
print("중복 데이터 개수 :", df['uniq_id'].duplicated(keep = False).sum())
```

### 결측 데이터 확인 및 처리
```python
#중복 데이터 제거
clean_data = df.drop_duplicates(subset=['uniq_id'], keep='first')

#review_title 결측값 채우기
clean_data['review_title'] = clean_data['review_title'].fillna("")

#review_text가 결측값 채우기
clean_data['review_text'] = clean_data['review_text'].fillna("")
```

### 데이터 확인
```python
#결측값 재확인
clean_data.isnull().sum()
```
<img width="146" alt="image" src="https://github.com/user-attachments/assets/92d3fe6e-0b20-479e-b560-b669b82fd860" />


### 정규 표현식을 사용한 특수 문자 제거
- `review_title`과 `review_text`를 대상으로 감성분석을 진행할 예정이므로 두 개에 대해 특수문자를 제거합니다.
```python
#노이즈(특수문자) 제거
def remove(text):
    clean = re.sub(r'[^A-Za-z\s]', '', text)
    return clean

clean_data['review_title'] = clean_data['review_title'].apply(remove)
clean_data['review_text'] = clean_data['review_text'].apply(remove)
```

### 데이터 토큰화
- `review_title`과 `review_text`에 대해 토큰화를 진행합니다.
```python
#토큰화
def token(text):
    tokens = word_tokenize(text)
    return tokens

clean_data['review_title'] = clean_data['review_title'].apply(token)
clean_data['review_text'] = clean_data['review_text'].apply(token)
```
<img width="978" alt="image" src="https://github.com/user-attachments/assets/3646102e-b473-4a8d-b7aa-9eaf57bf0f4d" />

### 불용어 처리
- `review_title`과 `review_text`에 대해 불용어 처리를 진행합니다.
```python
#불용어 제거
def stop(text):
    stopword = [word for word in text if word not in stopwords.words("english")]
    return stopword

clean_data['review_title'] = clean_data['review_title'].apply(stop)
clean_data['review_text'] = clean_data['review_text'].apply(stop)
```
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/6e721a9f-593f-4c4b-82be-6883874d503e" />

### 표제어 추출 및 스테밍
- 표제어 추출은 문맥에 따른 변형을 처리하여 단어의 기본 형태를 반환하고, 스테밍은 단어의 접미사나 변형을 규칙적으로 제거하여 어근을 반환합니다.
- 둘 중 하나만 진행해도 되지만, 두 개 다 처리하면 일반화를 더 잘 할 수 있을 것이라고 생각하여 둘 다 진행하였습니다.
```python
# 정규화 함수 정의
def lemmatize_and_stem(tokens):
    lemmatizer = WordNetLemmatizer()  # 표제어 추출을 위한 Lemmatizer 객체 생성
    stemmer = PorterStemmer()  # 스테밍을 위한 Stemmer 객체 생성
    
    # 표제어 추출
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # 스테밍
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
    
    return ' '.join(stemmed_tokens)  # 처리된 토큰들을 공백으로 결합하여 반환

# 데이터 프레임의 각 텍스트 필드에 대해 정규화 적용
clean_data['review_title'] = clean_data['review_title'].apply(lemmatize_and_stem)
clean_data['review_text'] = clean_data['review_text'].apply(lemmatize_and_stem)
```
<img width="962" alt="image" src="https://github.com/user-attachments/assets/b410efa8-8b6c-46c4-b37d-bdeae8a0b55d" />

## 3. 모델 구현
- VADER을 적용

### VADER 감정 분석기 초기화
```python
vader_sentiment = SentimentIntensityAnalyzer()

# 리뷰 텍스트에 대한 감정 점수를 계산하는 함수 정의
def calc_sentiment(review):
    # VADER 감정 분석기를 사용하여 감정 점수를 계산하고, 그 중에서 'compound' 점수를 반환
    return vader_sentiment.polarity_scores(review)["compound"]
```

### 감정 계산
```python
print("감정 계산 시작")
start = time.time()

# VADER를 사용하여 감성 점수 계산
clean_data["new_title"] = clean_data.review_title.apply(calc_sentiment)
clean_data["new_text"] = clean_data.review_text.apply(calc_sentiment)

end = time.time()
print("감정 계산 끝 " + str(round(end - start, 2)) + " 초 동안 진행")
```
### 감정 계산 결과
<img width="1086" alt="image" src="https://github.com/user-attachments/assets/a72d4ed4-5d04-43d1-b4ba-35b94acc4ee5" />


## 4. 모델 학습 및 튜닝
- 그리드 서치를 활용하여 하이퍼파라미터 튜닝을 진행하는데 교차 검증을 진행하여 견고성을 보장

### 데이터 벡터화 진행 및 데이터 분할
- 위에서 가장 잘 예측한 것이 review_text기 때문에 데이터는 review_text를 사용하였습니다.
- 벡터화를 진행하고 데이터를 분할하였습니다. 테이터의 비율은 train, test 8:2로 분할하였습니다.

```python
# 데이터 지정
y = clean_data["new_text_data"]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features = 1000) 
X = vectorizer.fit_transform(clean_data['review_text'])

# 학습 데이터와 테스트 데이터로 분할 진행
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
```
### 하이퍼 파라미터 튜닝 결과
로지스틱 회귀 모델을 만들고 그리드서치를 사용하여 하이퍼 파리미터 튜닝을 진행합니다.
교차 검증은 5로 설정합니다.
  
```python
# 필요한 라이브러리 임포트
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 로지스틱 회귀 모델 초기화
model = LogisticRegression()

# 하이퍼파라미터 그리드 설정
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# 그리드 서치 객체 생성 (모델, 하이퍼파라미터 그리드, 교차 검증 횟수, 평가 지표 설정)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")

# 그리드 서치 실행 (훈련 데이터에 맞추어 모델 학습)
grid_search.fit(X_train, y_train)

# 테스트 데이터에 대한 최적 모델의 성능 평가
score = grid_search.score(X_test, y_test)

# 최적 하이퍼파라미터와 테스트 정확도 출력
print("Best parameters:", grid_search.best_params_)
print("Test accuracy:", score)
```

하이퍼 파라미터 튜닝 결과는 아래와 같습니다.

1. 최적 하이퍼파라미터
- 그리드 서치가 선택한 최적의 하이퍼파라미터는 C = 10 입니다.
- C는 로지스틱 회귀 모델의 규제 강도를 제어하는 하이퍼파라미터로, 값이 클수록 규제가 약해지고, 값이 작을수록 규제가 강해지는데, 10을 통해 규제가 약간 작용하면서도 모델이 잘 학습되었다는 것을 알 수 있습니다.

2. 테스트 정확도
- 모델의 정확도는 약 92.63%로 높은 성능을 갖고 있다는 것을 알 수 있습니다.
<img width="267" alt="image" src="https://github.com/user-attachments/assets/e9578b71-d47e-4ad9-8306-c6ab199d58ab" />

### 모델 학습
- 최적의 모델을 사용하여 데이터 학습 진행
```python
final_model = grid_search.best_estimator_  # 그리드 서치를 통해 찾은 최적의 모델 추출
final_model.fit(X_train, y_train)  # 훈련 데이터를 사용하여 최적 모델 학습
y_pred = final_model.predict(X_test)  # 테스트 데이터에 대한 예측값 생성
```

## 5. 모델 평가
```python
# 정확도 평가
accuracy = round(accuracy_score(y_test, y_pred), 3)
print("Accuracy:", accuracy)

# 정밀도 평가
precision = round(precision_score(y_test, y_pred, average='weighted'), 3)
print("\nPrecision:", precision)

# 재현율 평가
recall = round(recall_score(y_test, y_pred, average='weighted'), 3)
print("\nRecall:", recall)

# F1 점수 평가
f1 = round(f1_score(y_test, y_pred, average='weighted'), 3)
print("\nF1 Score:", f1)

# ROC-AUC 점수 계산
roc_auc = round(roc_auc_score(y_test, y_pred), 3)
print("\nROC-AUC Score:", roc_auc)

#Classification report
class_report = classification_report(y_test, y_pred)
print("\n\t\t<Classification Report>\n")
print(class_report)
```
<img width="404" alt="image" src="https://github.com/user-attachments/assets/4e2d9216-98c2-4c89-ab93-8619d8d7a62d" />

1. Accuracy (정확도): 0.93의 높은 수치로, 모델이 대부분의 경우 올바르게 예측했다는 것을 알 수 있습니다.<br>

2. Precision (정밀도): 0.93의 높은 수치로, 모델이 양성 클래스를 예측할 때 대부분 올바르게 예측했다는 것을 알 수 있습니다.<br>

3. Recall (재현율): 0.93의 높은 수치로, 모델이 실제 양성 중 모델이 양성으로 올바르게 예측했다는 것을 알 수 있습니다.<br>

4. F1 Score: 0.93의 높은 수치로, 모델이 양성과 음성 예측 모두에서 균형 잡힌 성능을 보이고 있다는 것을 알 수 있습니다.<br>

5. ROC-AUC Score: 0.926로, 1에 가까운 수치인데, 이는 모델이 양성과 음성을 잘 구분하고 있다는 것을 알 수 있습니다.<br>

6. Classification Report: 모델은 전체적으로 높은 성능(정확도 93%, 정밀도 93%, 재현율 93%, F1 스코어 93%)을 보이며, 긍정 리뷰(94% 정밀도, 93% 재현율)와 부정 리뷰(91% 정밀도, 92% 재현율) 모두에서 균형 잡힌 예측을 하고 있습니다.<br>

7. 오버 피팅: 현재 모델의 정확도가 93%로 매우 높고, 다른 성능 지표들도 균형 잡혀 있습니다. 모델의 성능이 테스트 데이터에서도 높게 유지되므로, 오버피팅이 없다고 볼 수 있습니다.<br>

8. 언더 피팅: 현재 모델의 성능 지표(정확도, 정밀도, 재현율, F1 점수)가 모두 높습니다. 모델이 데이터를 잘 학습하고, 테스트 데이터에서도 좋은 성능을 보임으로써, 언더피팅 문제는 없음을 확인할 수 있습니다.<br>

`이를 통해 모델은 매우 우수한 성능을 갖고 있다고 볼 수 있습니다.`<br>


## 6. 시각화 및 보고

### Confusion Matrix
![image](https://github.com/user-attachments/assets/62d921e2-a078-423d-8618-12bb4a5a15b3)
1. True Negative (0.92)
- 모델이 부정 리뷰를 부정으로 정확히 예측한 비율입니다.
- 모델이 부정 리뷰를 잘 분류하고 있음을 나타냅니다.

2. False Positive (0.075)
- 모델이 부정 리뷰를 긍정으로 잘못 예측한 비율입니다.
- 7.5%의 부정 리뷰가 잘못 분류되었다는 것을 알 수 있습니다.

3. False Negative (0.072)
- 모델이 긍정 리뷰를 부정으로 잘못 예측한 비율입니다.
- 긍정 리뷰의 7.2%가 잘못 분류되었으며, 재현율이 높다는 것을 나타냅니다.

4. True Positive (0.93)
- 모델이 긍정 리뷰를 긍정으로 정확히 예측한 비율입니다.
- 긍정 리뷰의 93%가 정확하게 분류되었다는 것을 알 수 있습니다.

<br>

- 모델의 정확도는 약 92.5%로, 전반적인 성능이 매우 좋습니다.
- 정밀도는 92.5%로, 모델이 긍정 리뷰로 예측한 것 중 대부분이 실제 긍정 리뷰임을 나타냅니다.
- 재현율은 92.9%로, 모델이 실제 긍정 리뷰를 거의 모두 정확히 예측하고 있음을 보여줍니다.
- F1 점수는 92.7%로, 정밀도와 재현율의 균형을 나타내며, 모델의 전반적인 성능을 종합적으로 평가합니다.

`모델은 부정 리뷰와 긍정 리뷰 모두에서 충분히 높은 성능을 갖고 있기 때문에 성능 개선을 필요성이 낮다고 볼 수 있습니다.`

### ROC 곡선
![image](https://github.com/user-attachments/assets/da6e8772-b88f-43d7-a8e7-1adea578f3fc)

ROC 곡선이 대각선 기준선보다 위에 위치할수록 모델의 성능이 우수함을 나타내는데, 주황색 실선이 대각선보다 위에 있으므로, 모델이 무작위 분류기보다 성능이 좋음을 알 수 있습니다.

AUC는 1에 가까울수록 성능이 좋다는 것을 의미하는데, 0.926이라는 것을 통해 모델이 분류를 잘 한다는 것을 알 수 있습니다.

`즉, ROC 곡선과 AUC 값 모두 높은 값을 가지고 있으므로 모델의 성능이 매우 좋다는 것을 알 수 있습니다.`
