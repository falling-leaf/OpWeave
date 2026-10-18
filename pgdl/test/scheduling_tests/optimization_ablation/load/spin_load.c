#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <time.h>
static volatile int stop;
static void on_term(int sig) { (void)sig; stop = 1; }
int main(int argc, char **argv) {
    int cpu = argc > 1 ? atoi(argv[1]) : 0;
    int duty = argc > 2 ? atoi(argv[2]) : 100;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
    signal(SIGTERM, on_term);
    struct timespec pause = {0, 1000000L};  /* 1 ms */
    unsigned acc = 0;
    while (!stop) {
        for (int i = 0; i < duty && !stop; i++) {
            for (int k = 0; k < 20000; k++) acc = acc * 1664525u + 1013904223u;
        }
        if (duty < 100) nanosleep(&pause, NULL);
    }
    return (int)acc;
}
