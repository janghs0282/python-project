import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


#데이터 불러오기
train = pd.read_csv('titanic.csv')

#데이터 결측치 예측
train.isnull().sum()

#상위 5개 항목 출력
train.head()

#데이터 분포(평균, 최소값, 최대값 등)를 살펴보기
train.describe()

#각 컬럼 데이터 타입살펴보기
train.dtypes

# Cabin과 Embraked의 단일값들 확인
train.Cabin.unique()
train.Embarked.unique()

#생존자와 사망자 확인
survival = train.Survived.sum() # 생존자
n_survival = train.shape[0] - survival # 전체 - 생존자 = 사망자

#각 클래스별 탑승객 분포 확인
train['Pclass'].value_counts()

#성별 분포
train['Sex'].value_counts()

#'Embarked' 필드의 결측치는 값이 가장 많은 'S'로 할당하기
train['Embarked'] = train['Embarked'].fillna('S')

#'Age' 필드의 결측치는 값이 중간값으로 할당하기
train['Age'] = train['Age'].fillna(train['Age'].median())

#'Name'필드에서 신분을 나타내는 단어를 뽑아서 'Title' 필드에 할당하기
train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)

# 'Title' 필드의 카테고리를 6개의 항목으로 변경하기
train['Title'] = train['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')
train['Title'] = train['Title'].replace(['Jonkheer', 'Master'], 'Master')
train['Title'] = train['Title'].replace(['Don', 'Sir', 'the Countess', 'Lady', 'Dona'], 'Royalty')
train['Title'] = train['Title'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs')
train['Title'] = train['Title'].replace(['Mlle', 'Miss'], 'Miss')
train['Title'] = train['Title'].replace(['Mr'], 'Mr')

#변수 y를 선언해서 학습할 목표변수(=종속변수)인 'Survived'필드값을 담기
y = train.Survived

#나이('Age') 필드를 그룹핑하여 'AgeGroup'필드 생성하여 할당하기
bin = [0, 18, 25, 35, 60, 100]
group_names = ['Baby', 'Youth', 'YoungAdult', 'MiddleAged', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins=bin, labels=group_names)
train['AgeGroup'].value_counts()

#성별('Sex')의 생존여부('Survived') 데이터 분포확인하기 : barplot
sns.barplot(x=train['Sex'], y=train['Survived'], hue=train['Sex'], dodge=False)
#plt.show()
#연령분포('AgeGroup')별 + 클래스('Pclass')별 생존여부('Survived') 데이터 분포확인하기: barplot
sns.barplot(x='AgeGroup', y='Survived', hue='Pclass', data=train)

#전체 변수의 correlation에 대해 히트맵 그리기
plt.subplots(figsize=(8,6))
sns.heatmap(train.corr(), annot=True, linewidths=2)
#plt.show()

# 'Name', 'Ticket', 'SibSp', 'Parch', 'Cabin' 컬럼 삭제하기
train = train.drop(['Name', 'Ticket', 'SibSp', 'Parch', 'Cabin'], axis=1)

# 모델링에 사용할 변수의 타입을 숫자로 변환
train['Sex'].dtypes
train['Sex'] = train['Sex'].astype(str)

label = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title', 'AgeGroup']:
    train[col] = label.fit_transform(train[col])

#학습시킬 변수와 Label변수를 분리.
X_train = train[['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'AgeGroup']]
Y_train = train[['Survived']]

lr = LogisticRegression()

#테스트 데이터 로드
test = pd.read_csv('titanictest.csv')

# 널값확인
print(test.isnull().sum())

# age 와 Fare 중앙값으로 대체
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

# train 데이터셋과 차원 맞춰주기
train = train.drop(['PassengerId','Age','Embarked', 'Title', 'AgeGroup'], axis=1)
test = test.drop(['Age', 'PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Cabin', 'Embarked'], axis=1)

# 'Sex'컬럼을 Object형에서 Interger(Number)형으로 변환해주기.
test['Sex'] = label.fit_transform(test['Sex'])

print(test.head())
print(train.head())

#Logistic Regression모델로 예측하기.
pred = lr.predict(test)
