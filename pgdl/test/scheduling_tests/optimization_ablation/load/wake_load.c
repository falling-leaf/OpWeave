#define _GNU_SOURCE
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
static volatile int stop;
static void on_term(int sig) { (void)sig; stop = 1; }
static void futex_wait(volatile int *addr, int val) {
    syscall(SYS_futex, addr, FUTEX_WAIT, val, NULL, NULL, 0);
}
static void futex_wake(volatile int *addr) {
    syscall(SYS_futex, addr, FUTEX_WAKE, 1, NULL, NULL, 0);
}
typedef struct { volatile int *mine; volatile int *peer; int gap; } pair_t;
static void *ping(void *arg) {
    pair_t *p = arg;
    while (!stop) {
        __sync_lock_test_and_set(p->mine, 1);
        futex_wake(p->mine);
        for (volatile int i = 0; i < p->gap; i++) {}
        while (!__sync_lock_test_and_set(p->peer, 0) && !stop) futex_wait(p->peer, 0);
    }
    futex_wake(p->peer);
    return NULL;
}
int main(int argc, char **argv) {
    int cpu = argc > 1 ? atoi(argv[1]) : 20;
    int gap = argc > 2 ? atoi(argv[2]) : 0;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
    signal(SIGTERM, on_term);
    volatile int flags[2] = {0, 0};
    pair_t a = {&flags[0], &flags[1], gap};
    pair_t b = {&flags[1], &flags[0], gap};
    pthread_t tid;
    pthread_create(&tid, NULL, ping, &b);
    ping(&a);
    pthread_join(tid, NULL);
    return 0;
}
