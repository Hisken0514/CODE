#include <bits/stdc++.h>
using namespace std;

string optable[32][2];
string F22[100][100];
string F23[100][100];
string F26[100][100];

void RDfile(string F22[100][100],string a);
void WRfile(string F22[100][100],string a);
void LDoptable();
void FixFugure(string F22[100][100]);
void pass1(string F22[100][100]);
void pass2(string F22[100][100]);
void F23figure();

string toHex(int v);
int to10(string v);

int main() {
    LDoptable();
    RDfile(F26,"Figure2.5.txt");
    pass1(F26);
    pass2(F26);
    cout <<"Figure2.6.txt"<<endl<<endl;
    for (int i = 0; i < 100; i++) {
        if(F26[i][1] == "" && F26[i][2] == "")continue;
        for (int j = 0; j < 5; j++) {
             cout << F26[i][j] <<"\t";
        }
         cout << endl;
    }
    cout <<endl;
    system("PAUSE");
    return 0;
}
void RDfile(string F22[100][100],string a) {
    ifstream file(a);
    if (!file.is_open()) {
        cerr << "Unable to open file." << endl;
        return;
    }

    string line;
    int index = 0;
    while (getline(file, line) && index < 100) {
        istringstream iss(line);
        int col = 1;
        while (iss >> F22[index][col]) {
            if(col > 3 || F22[index][col - 1] == "SUBROUTINE" ){
               F22[index][2] += " " + F22[index][col];
               F22[index][col] = "";
            }
            col++;
        }
        index++;
    }
    FixFugure(F22);
    file.close();
}
void WRfile(string F22[100][100] , string a){
    ofstream outFile(a);
     if (!outFile.is_open()) {
        cerr << "無法打開文件!" << endl;
    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            outFile << F22[i][j] << "\t";
        }
        outFile << endl;
    }
    outFile.close();
    cout << endl << a <<"\tfinish!";
}
void LDoptable() {
    ifstream file("optable.txt");
    if (!file.is_open()) {
        cerr << "Unable to open optable.txt" << endl;
        return;
    }

    string opcode, value;
    int index = 0;
    while (file >> opcode >> value && index < 32) {
        optable[index][0] = opcode;
        optable[index][1] = value;
        index++;
    }

    file.close();
}
void pass1(string F22[100][100]) {
    int t = stoi(F22[0][3]),x = 0; 
    for(int i = 0;i < 100 ;i++){
        if(x == 0 && F22[i][1] == "."){
            x++;
            t+=4096-3;
        }
        if(F22[i][2] == "START"){
            F22[i][0] = "0000";
            F22[i+1][0] = "0000";
            continue;
        }
        else if(F22[i][1] == "END"){
            break;
        }
        else if(F22[i][1] == "."){
            continue;
        }
        for(int j = 0;j < 32;j++){
            if(j < 29){
                if(F22[i][2] == optable[j][0] || F22[i][2][0] == '+'|| F22[i][2][0] == '@'|| F22[i][2][0] == '#' && j < 28){
                F22[i][0] = toHex(t);
                t += 3;
                if(F22[i][2][0] == '+')t++;
                break;
            }
            }
            
            if(F22[i][2] == optable[j][0]){
                F22[i][0] = toHex(t);
                t += 2;
                break;
            }
            else if(F22[i][1] == "INPUT"){
            F22[i][0] = toHex(t);
            t++;
            break;;
            }
            else if(F22[i][2] =="WORD" ||F22[i][2] =="BYTE" ||F22[i][2] =="RESW" ||F22[i][2] =="RESB"||F22[i][2] =="RSUB"){
            F22[i][0] = toHex(t);
            t += 3;
            break;
            }
        }
    }
}
void pass2(string F22[100][100]){
    for(int i = 0;i < 100 ;i++){
        if(F22[i][2] == "START"){
            continue;
        }
        else if(F22[i][1] == "RESW"){
            continue;
        }
        else if(F22[i][1] == "."){
            continue;
        }
        else if(F22[i][2] == "END"){
            continue;
        }
        else if(F22[i][2] == "WORD"){
            F22[i][4] = "00" + toHex(stoi(F22[i][3]));
        }
        else if(F22[i][2] == "BYTE"){
            int a = 0,t = 0;
            while(a != 2){
                if(F22[i][3][t] == '\''){
                    a++;
                    t++;
                    continue;
                }
                if(a == 1)F22[i][4] += F22[i][3][t];
                if(F22[i][4] == "EOF")F22[i][4] = "454f46";
                t++;
            }
            continue;
        }
        for(int j = 0;j < 32;j++){
            if(F22[i][2] == optable[j][0] ){
                F22[i][4] = optable[j][1];
                if(F22[i][3][0] == '#'){
                    F22[i][4][1] += 1;
                }
                if(F22[i][2][0] == '+'){

                }
                if(F22[i][3][0] == '@'){
                    F22[i][4][1] += 2;
                }
                if(F22[i][3][0] != '@' && F22[i][3][0] != '#'){
                    if(F22[i][2] == "CLEAR")break;
                    F22[i][4][1] += 3;
                }
                break;
            }
        }
        for(int x = 0;x < 100;x++){
                if(F22[i][3] == F22[x][1]){
                    F22[i][4] += F22[x][0];
                    break;
                }
        }
        if(F22[i][3] == "RSUB"){
            F22[i][4] += "0000";
        }
        if(F22[i][3] == "A"){
            F22[i][4] += "00";
        }
        if(F22[i][3] == "X"){
            F22[i][4] += "10";
        }
        if(F22[i][3] == "S"){
            F22[i][4] += "40";
        }
        if(F22[i][3] == "T"){
            F22[i][4] += "50";
        }  
        if(F22[i][2] == "LDCH" || F22[i][2] == "STCH"){
            string cc;
            int a = 0,t = 0;
            while(a != 1 && cc.size() != F22[i][3].size()-2){
                if(F22[i][3][t] == ','){
                    a++;
                    t++;
                    break;
                }
                cc += F22[i][3][t];
                t++;
            }
            for(int k = 0; k < 100;k++){
                if(cc == F22[k][1]){
                    F22[i][4] += F22[k][0];
                    F22[i][4][2] += 8;
                    break;
                }
            }
            continue;
        }
    }
    F23figure();
}
void FixFugure(string F22[100][100]){
    for (int i = 0; i < 100; i++) {
            for(int j = 0;j < 32;j++){
               if(F22[i][1] == optable[j][0] || F22[i][0] == "END" || F22[i][1][0] == '+'|| F22[i][1][0] == '@'|| F22[i][1][0] == '#'|| F22[i][1] == "BASE"){
                F22[i][3] = F22[i][2];
                F22[i][2] = F22[i][1];
                F22[i][1] = " ";
            } 
        }
    }
}
void F23figure(){
    int a = 0,t = 1;
    for(int i = 0 ; i < 100 ;i++){
        if(F22[i][2] == "END" ){
            a = to10(F22[i - 1][0]);
        }
    }
    F23[0][0] = "HCOPY\t";
    F23[0][1] = "00" + F22[0][0] + "00" + toHex( a- to10("1000") + 1);
    for(int i = 1 ;i < 100 ;i++){
        if(F22[i][4] != ""){
            F23[t][1] += F22[i][4];
            if(F23[t][0] == ""){
                F23[t][0] = "T00" + F22[i][0];
            }
            if(F23[t][1].size() + F23[t][0].size() > 62 || F22[i][4] == "000000"){
                F23[t][0] += toHex(F23[t][1].size() / 2);
                t++;
            }
        }
        if(F22[i][2] == "END"){
                F23[t + 1][0] = "E00" + F22[0][0];
                break;
            }
    }
}
string toHex(int v) {
    int i = 3;
    string result ="0000";
    string c = "0123456789ABCDEF";
    while(v != 0 && i >= 0){
        int r = v % 16;
        v /= 16;
        result[i] = c[r];
        i--;
    }
    return result;
}
int to10(string v) {
    int result = 0;
    for(int i = 3;i >= 0;i--){
        if(v[i] >= 'A'){
            result += (v[i] - 'A' + 10) * pow(16, 3 - i);
        }
        else if ( v[i] <= '9') {
            result += (v[i] - '0') * pow(16, 3 - i);
        }
    }
    
    return result;
}

