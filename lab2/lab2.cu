// nvcc opengl_2.cu -lGL -lGLU -lglut -lGLEW
// ./a.out

// Для linux нужно поставить пакеты: libgl1-mesa-dev libglew-dev freeglut3-dev
// sudo apt-get install libgl1-mesa-dev libglew-dev freeglut3-dev

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>

typedef unsigned char uchar;
using namespace std;

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

#define sqr3(x) ((x)*(x)*(x))
#define sqr(x) ((x)*(x))

//Структура объекта
struct Item {
    float x;
    float y;
    float z;
    float dx;
    float dy;
    float dz;
    float angle;
    float q;
};

//Коэффициент замедления
const float slow = 0.9999;

//Время
const float dt = 0.005;

//Коэффициенты отталкивания
const float e_0 = 1e-3;
const float k = 150.0;

//Количество сфер
const int items_count = 100;

Item *items = (Item *)malloc(sizeof(Item) * items_count);
Item bullet;

//Скорость пули
double bullet_speed = 25;

//Размеры окна
int w = 1920, h = 1080; 

//Гравитация
const double gravity = 0.01; 



//Параметры игрока
    //Начальные координаты
    float player_x = -1.5, player_y = -1.5, player_z = 1.0;

    //Начальные скорости игрока
    float player_dx = 0.0, player_dy = 0.0, player_dz = 0.0;

    //Начальные углы
    float player_yaw = 0.0, player_pitch = 0.0;

    //Инерция углов
    float player_dyaw = 0.0, player_dpitch = 0.0;

    //Скороcть игрока
    float player_speed = 0.05;


//Параметры куба
    //Размер куба
    const float cub_size  = 30.0;

    // Размер текстуры пола          
    const int text_floor_size = 100;  

    //Объект для хранения буффера
    cudaGraphicsResource *res;  

    //Массив из текстурных номеров    
    GLuint textures[3];    

    //Номер буфера 
    GLuint vbo;         

    //Заряд стен куба
    const float q_cube = 1;                   

    //quadric объекты - это геометрические фигуры 2-го порядка, т.е. сфера, цилиндр, диск, конус. 
    GLUquadric* quadratic; 


__global__ void kernel_floor(uchar4 *data, Item bullet, Item *items, float t) {    // Генерация текстуры пола на GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int i, j;
    float x, y,fb;
    for(i = idx; i < text_floor_size; i += offsetx) {
        for(j = idy; j < text_floor_size; j += offsety) {
            fb = 0;
            x = (2.0 * i / (text_floor_size - 1.0) - 1.0) * cub_size;
            y = (2.0 * j / (text_floor_size - 1.0) - 1.0) * cub_size;

            for (int p = 0; p < items_count; ++p) {
                fb += 5 * k * items[p].q / (sqr(x - items[p].x) + sqr(y - items[p].y) + sqr(0.75 - items[p].z) + e_0);
            }

            fb += 5 * k * bullet.q / (sqr(x - bullet.x) + sqr(y - bullet.y) + sqr(0.75 - bullet.z) + e_0);
     
            fb = min(max(0.0f, fb), 255.0f);
            data[j * text_floor_size + i] = make_uchar4(100, 200, fb, 255);
        }
    }
}

__global__ void kernel_move(Item *items, Item bullet, float player_x, float player_y, float player_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (i < items_count) {

        //Гравитация
        items[i].dz -= gravity;

        // Замедление
        items[i].dx *= slow;
        items[i].dy *= slow;
        items[i].dz *= slow;

        // Отталкивание от стен
        items[i].dx += q_cube * items[i].q * k * (items[i].x - cub_size) / (sqr3(fabs(items[i].x - cub_size)) + e_0) * dt;
        items[i].dx += q_cube * items[i].q * k * (items[i].x + cub_size) / (sqr3(fabs(items[i].x + cub_size)) + e_0) * dt;

        items[i].dy += q_cube * items[i].q * k * (items[i].y - cub_size) / (sqr3(fabs(items[i].y - cub_size)) + e_0) * dt;
        items[i].dy += q_cube * items[i].q * k * (items[i].y + cub_size) / (sqr3(fabs(items[i].y + cub_size)) + e_0) * dt;

        items[i].dz += q_cube * items[i].q * k * (items[i].z - 2 * cub_size) / (sqr3(fabs(items[i].z - 2 * cub_size)) + e_0) * dt;
        items[i].dz += q_cube * items[i].q * k * (items[i].z + 0.0) / (sqr3(fabs(items[i].z + 0.0)) + e_0) * dt;

        // Отталкивание от камеры
        float l = sqrt(sqr(items[i].x - player_x) + sqr(items[i].y - player_y) + sqr(items[i].z - player_z));
        items[i].dx += 3.0 * items[i].q * k * (items[i].x - player_x) / (l * l * l + e_0) * dt;
        items[i].dy += 3.0 * items[i].q * k * (items[i].y - player_y) / (l * l * l + e_0) * dt;
        items[i].dz += 3.0 * items[i].q * k * (items[i].z - player_z) / (l * l * l + e_0) * dt;


        // Отталкивание от пули
        l = sqrt(sqr(items[i].x - bullet.x) + sqr(items[i].y - bullet.y) + sqr(items[i].z - bullet.z));
        items[i].dx += bullet.q * items[i].q * k * (items[i].x - bullet.x) / (l * l * l + e_0) * dt;
        items[i].dy += bullet.q * items[i].q * k * (items[i].y - bullet.y) / (l * l * l + e_0) * dt;
        items[i].dz += bullet.q * items[i].q * k * (items[i].z - bullet.z) / (l * l * l + e_0) * dt;

        //Отталкивание от других сфер
        for (int j = 0; j < items_count; ++j) {

            if (i == j) {
                continue;
            }

            l = sqrt(sqr(items[i].x - items[j].x) + sqr(items[i].y - items[j].y) + sqr(items[i].z - items[j].z));
            items[i].dx += items[i].q * items[i].q * k * (items[i].x - items[j].x) / (l * l * l + e_0) * dt;
            items[i].dy += items[i].q * items[i].q * k * (items[i].y - items[j].y) / (l * l * l + e_0) * dt;
            items[i].dz += items[i].q * items[i].q * k * (items[i].z - items[j].z) / (l * l * l + e_0) * dt;
        }

        i += offset;
    }
}

__global__ void kernel_res(Item *items) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (i < items_count) {
        items[i].x += items[i].dx * dt;
        items[i].y += items[i].dy * dt;
        items[i].z += items[i].dz * dt;

        i += offset;
    }
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0f, (GLfloat)w/(GLfloat)h, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(player_x, player_y, player_z,
                player_x + cos(player_yaw) * cos(player_pitch),
                player_y + sin(player_yaw) * cos(player_pitch),
                player_z + sin(player_pitch),
                0.0f, 0.0f, 1.0f);

    glBindTexture(GL_TEXTURE_2D, textures[2]);
    for (int i = 0; i < items_count; ++i) {
        glPushMatrix();
        glTranslatef(items[i].x, items[i].y, items[i].z);
        glRotatef(items[i].angle, 0.0, 0.0, 1.0);
        gluSphere(quadratic, 1.5f, 16, 16);
        glPopMatrix(); 
        items[i].angle += 0.25;
    }   

    glBindTexture(GL_TEXTURE_2D, textures[1]);

    glPushMatrix();
    glTranslatef(bullet.x, bullet.y, bullet.z);
    gluSphere(quadratic, 1.0f, 16, 16);
    glPopMatrix(); 


    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)text_floor_size, (GLsizei)text_floor_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    glBegin(GL_QUADS);   
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-cub_size, -cub_size, 0.0);

        glTexCoord2f(1.0, 0.0);
        glVertex3f(cub_size, -cub_size, 0.0);

        glTexCoord2f(1.0, 1.0);
        glVertex3f(cub_size, cub_size, 0.0);

        glTexCoord2f(0.0, 1.0);
        glVertex3f(-cub_size, cub_size, 0.0);
    glEnd();

    
    glBindTexture(GL_TEXTURE_2D, 0);
           
    glLineWidth(2);            
    glColor3f(0.5f, 0.5f, 0.5f);
    glBegin(GL_LINES); 
        glVertex3f(-cub_size, -cub_size, 0.0);
        glVertex3f(-cub_size, -cub_size, 2.0 * cub_size);

        glVertex3f(cub_size, -cub_size, 0.0);
        glVertex3f(cub_size, -cub_size, 2.0 * cub_size);

        glVertex3f(cub_size, cub_size, 0.0);
        glVertex3f(cub_size, cub_size, 2.0 * cub_size);

        glVertex3f(-cub_size, cub_size, 0.0);
        glVertex3f(-cub_size, cub_size, 2.0 * cub_size);
    glEnd();


    glBegin(GL_LINE_LOOP);                      
        glVertex3f(-cub_size, -cub_size, 0.0);
        glVertex3f(cub_size, -cub_size, 0.0);
        glVertex3f(cub_size, cub_size, 0.0);
        glVertex3f(-cub_size, cub_size, 0.0);
    glEnd();

    glBegin(GL_LINE_LOOP);
        glVertex3f(-cub_size, -cub_size, 2.0 * cub_size);
        glVertex3f(cub_size, -cub_size, 2.0 * cub_size);
        glVertex3f(cub_size, cub_size, 2.0 * cub_size);
        glVertex3f(-cub_size, cub_size, 2.0 * cub_size);
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f);

    glutSwapBuffers();
}



void update() {

    //Гравитация для игрока
    //player_dz -= gravity; 

    float player_v = sqrt(player_dx * player_dx + player_dy * player_dy + player_dz * player_dz);


    // Ограничение максимальной скорости
    if (player_v > player_speed) {        
        player_dx *= player_speed / player_v;
        player_dy *= player_speed / player_v;
        player_dz *= player_speed / player_v;
    }
    player_x += player_dx; player_dx *= 0.99;
    player_y += player_dy; player_dy *= 0.99;
    player_z += player_dz; player_dz *= 0.99;

    // Пол, ниже которого камера не может переместиться
    if (player_z < 1.0) {          
        player_z = 1.0;
        player_dz = 0.0;
    }

    // Вращение камеры
    if (fabs(player_dpitch) + fabs(player_dyaw) > 0.0001) {   
        player_yaw += player_dyaw;
        player_pitch += player_dpitch;
        player_pitch = min(M_PI / 2.0 - 0.0001, max(-M_PI / 2.0 + 0.0001, player_pitch));
        player_dyaw = player_dpitch = 0.0;
    }

    Item *dev_items;
    CSC(cudaMalloc(&dev_items, sizeof(Item) * items_count));
    CSC(cudaMemcpy(dev_items, items, sizeof(Item) * items_count, cudaMemcpyHostToDevice));

    kernel_move<<<128, 128>>>(dev_items, bullet, player_x, player_y, player_z);

    kernel_res<<<128, 128>>>(dev_items);


    static float t = 0.0;
    uchar4* dev_data;
    size_t size;
    cudaGraphicsMapResources(1, &res, 0);
    cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res);
    kernel_floor<<<dim3(32, 32), dim3(32, 8)>>>(dev_data, bullet, dev_items, t);      
    cudaGraphicsUnmapResources(1, &res, 0);

    CSC(cudaMemcpy(items, dev_items, sizeof(Item) * items_count, cudaMemcpyDeviceToHost));

    bullet.x += bullet.dx * dt;
    bullet.y += bullet.dy * dt;
    bullet.z += bullet.dz * dt;

    
    t += 0.005;

    glutPostRedisplay();    
}


//Обработка кнопок
void keys(unsigned char key, int x, int y) {    
    switch (key) {
        //"W" Движение вперед
        case 'w':                 
            player_dx += cos(player_yaw) * cos(player_pitch) * player_speed;
            player_dy += sin(player_yaw) * cos(player_pitch) * player_speed;
            player_dz += sin(player_pitch) * player_speed;
        break;

        //"S" Назад
        case 's':                 
            player_dx += -cos(player_yaw) * cos(player_pitch) * player_speed;
            player_dy += -sin(player_yaw) * cos(player_pitch) * player_speed;
            player_dz += -sin(player_pitch) * player_speed;
        break;

        //"A" Влево
        case 'a':                 
            player_dx += -sin(player_yaw) * player_speed;
            player_dy += cos(player_yaw) * player_speed;
            break;
            
        //"D" Вправо
        case 'd':                 
            player_dx += sin(player_yaw) * player_speed;
            player_dy += -cos(player_yaw) * player_speed;
        break;

        //"ESC" выход
        case 27:
            cudaGraphicsUnregisterResource(res);
            glDeleteTextures(3, textures);
            glDeleteBuffers(1, &vbo);
            gluDeleteQuadric(quadratic);
            exit(0);
        break;
    }
}

void mouse(int x, int y) {
    static int x_prev = w / 2, y_prev = h / 2;
    float dx = 0.005 * (x - x_prev);
    float dy = 0.005 * (y - y_prev);
    player_dyaw -= dx;
    player_dpitch -= dy;
    x_prev = x;
    y_prev = y;

    // Перемещаем указатель мышки в центр, когда он достиг границы
    if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) {
        glutWarpPointer(w / 2, h / 2);
        x_prev = w / 2;
        y_prev = h / 2;
    }
}

void reshape(int w_new, int h_new) {
    w = w_new;
    h = h_new;
    glViewport(0, 0, w, h);                                 
    glMatrixMode(GL_PROJECTION);                        
    glLoadIdentity();                                       
}

void shoot(int button, int state, int x, int y) {
    if (state == GLUT_UP && button == GLUT_LEFT_BUTTON) {
        bullet.x = player_x;// + cos(player_yaw) * cos(player_pitch) * bullet_speed;
        bullet.y = player_y;// + sin(player_yaw) * cos(player_pitch) * bullet_speed;
        bullet.z = player_z;// + sin(player_pitch) * bullet_speed ;
        bullet.dx = cos(player_yaw) * cos(player_pitch) * bullet_speed;
        bullet.dy = sin(player_yaw) * cos(player_pitch) * bullet_speed;
        bullet.dz = sin(player_pitch) * bullet_speed;
    }
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );
    glutInitWindowSize(w, h);
    glutCreateWindow("Lab");

    glutIdleFunc(update);
    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutPassiveMotionFunc(mouse);
    glutMouseFunc(shoot);
    glutReshapeFunc(reshape);

    glutSetCursor(GLUT_CURSOR_NONE); 

    int wt, ht;
    FILE *in = fopen("goose.data", "rb");
    fread(&wt, sizeof(int), 1, in);
    fread(&ht, sizeof(int), 1, in);
    uchar *data = (uchar *)malloc(sizeof(uchar) * wt * ht * 4);
    fread(data, sizeof(uchar), 4 * wt * ht, in);
    fclose(in);

    glGenTextures(3, textures);
    glBindTexture(GL_TEXTURE_2D, textures[2]);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //GL_LINEAR);        
    
    free(data);
    in = fopen("bullet.data", "rb");
    fread(&wt, sizeof(int), 1, in);
    fread(&ht, sizeof(int), 1, in);
    data = (uchar *)malloc(sizeof(uchar) * wt * ht * 4);
    fread(data, sizeof(uchar), 4 * wt * ht, in);
    fclose(in);

    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 

    free(data);

    quadratic = gluNewQuadric();
    gluQuadricTexture(quadratic, GL_TRUE);  

    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);   // Интерполяция 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);   // Интерполяция 

    glBindTexture(GL_TEXTURE_2D, 0);


    glEnable(GL_TEXTURE_2D);                           
    glShadeModel(GL_SMOOTH);                             
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);                
    glClearDepth(1.0f);                                 
    glDepthFunc(GL_LEQUAL);                            
    glEnable(GL_DEPTH_TEST);                             
    glEnable(GL_CULL_FACE);                             


    glewInit();                     
    glGenBuffers(1, &vbo);                        
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);          
    glBufferData(GL_PIXEL_UNPACK_BUFFER, text_floor_size * text_floor_size * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard); 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);           


    bullet.x = 100;
    bullet.y = 100;
    bullet.z = 100;
    bullet.q = 10;


    double step = 5;
    double x = -cub_size + 1;
    double y = -cub_size + 1;
    double z = cub_size - 1;

    for (int i = 0; i < items_count; ++i) {

        items[i].angle = i / (float)items_count * 2 * 3.14 * 30;

        items[i].x = x;
        items[i].y = y;
        items[i].z = z; 

        items[i].dx = items[i].dy = items[i].dz = 0;
        items[i].q = 1.0;

        x += step;

        if (x >= cub_size) {
            x = -cub_size + 1;
            y += step;
        }

        if (y >= cub_size) {    
            x = -cub_size + 1;
            y = -cub_size + 1;
            z -= step;
        }
    }


    glutMainLoop();
}
